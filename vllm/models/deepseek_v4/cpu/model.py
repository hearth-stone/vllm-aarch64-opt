# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek V4 model implementation for CPU."""

import typing
from collections.abc import Callable, Iterable
from itertools import islice

import regex as re
import torch
import torch.nn as nn

from vllm import envs
from vllm.config import VllmConfig
from vllm.distributed import (
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.activation import SiluAndMul, SiluAndMulWithClamp
from vllm.model_executor.layers.fused_moe import (
    FusedMoEFactory,
    fused_moe_make_expert_params_mapping,
)
from vllm.model_executor.layers.fused_moe.router.gate_linear import GateLinear
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.interfaces import (
    EagleModelMixin,
    MixtureOfExperts,
    SupportsEagle3,
    SupportsPP,
)
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    WeightsMapper,
    extract_layer_index,
    is_pp_missing_parameter,
    make_layers,
    maybe_prefix,
)
from vllm.sequence import IntermediateTensors
from vllm.utils.math_utils import cdiv
from vllm.v1.utils import record_function_or_nullcontext

from .attention import DeepseekV4CPUAttention
from .mhc import (
    broadcast_residual,
    fused_cpp_mhc_post_head_rmsnorm,
    fused_cpp_mhc_post_pre_rmsnorm,
    fused_cpp_mhc_pre_rmsnorm,
    hc_head,
    mhc_fused_post_pre,
    mhc_post,
    mhc_pre,
    prepare_fused_cpp_mhc_weight,
)

logger = init_logger(__name__)


def _configure_cpu_router_gate(gate: GateLinear, *, strict: bool) -> None:
    gate._cpu_fused_cpp_linear_enabled = True
    gate._cpu_fused_cpp_linear_required = strict
    gate._cpu_fused_cpp_out_dtype = torch.float32


class DeepseekV4MLP(nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        swiglu_limit: float | None,
        quant_config: QuantizationConfig | None,
        prefix: str,
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size, intermediate_size],
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.down_proj",
        )
        for projection in (self.gate_up_proj, self.down_proj):
            projection._cpu_keep_raw_weight = True
            projection._cpu_combined_moe_owned = True
            projection._cpu_fused_cpp_joint_int8_owned = True
        if hidden_act != "silu":
            raise ValueError(f"Unsupported DeepSeek V4 activation: {hidden_act}")
        self.act_fn = (
            SiluAndMulWithClamp(swiglu_limit)
            if swiglu_limit is not None
            else SiluAndMul()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up = self.gate_up_proj(x)[0]
        return self.down_proj(self.act_fn(gate_up))[0]


class DeepseekV4MoE(nn.Module):
    def __init__(self, vllm_config: VllmConfig, prefix: str) -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        parallel_config = vllm_config.parallel_config
        self.prefix = prefix
        self.use_mega_moe = False
        self.tp_size = get_tensor_model_parallel_world_size()
        self.routed_scaling_factor = getattr(config, "routed_scaling_factor", 1.0)
        self.n_routed_experts = config.n_routed_experts
        self.n_activated_experts = config.num_experts_per_tok
        self.n_shared_experts = config.n_shared_experts or 0
        self.n_logical_experts = self.n_routed_experts
        self.n_redundant_experts = parallel_config.eplb_config.num_redundant_experts
        self.n_physical_experts = self.n_logical_experts + self.n_redundant_experts
        self.n_local_physical_experts = self.n_physical_experts
        self.swiglu_limit = config.swiglu_limit
        self.scoring_func = getattr(config, "scoring_func", "sqrtsoftplus")

        if vllm_config.kernel_config.moe_backend == "deep_gemm_mega_moe":
            raise NotImplementedError(
                "deep_gemm_mega_moe is a GPU backend; use auto on CPU"
            )
        if parallel_config.enable_expert_parallel:
            raise NotImplementedError(
                "DeepSeek V4 fused_cpp Plan V2 supports TP on CPU, not expert parallel"
            )

        self.gate = GateLinear(
            input_size=config.hidden_size,
            output_size=config.n_routed_experts,
            bias=False,
            out_dtype=torch.float32,
            prefix=f"{prefix}.gate",
        )
        # The current CPU baseline computes router logits with fused_cpp and
        # returns fp32.  Keep that numerically stable path for V4: small BF16
        # GEMM differences can otherwise change the discrete expert selection.
        _configure_cpu_router_gate(self.gate, strict=envs.VLLM_CPU_FUSED_CPP_STRICT)
        self.gate.e_score_correction_bias = None
        self.gate.tid2eid = None
        if extract_layer_index(prefix) < config.num_hash_layers:
            self.gate.tid2eid = nn.Parameter(
                torch.randint(
                    0,
                    config.n_routed_experts,
                    (config.vocab_size, config.num_experts_per_tok),
                    dtype=torch.int32,
                ),
                requires_grad=False,
            )
        elif getattr(config, "topk_method", None) == "noaux_tc":
            self.gate.e_score_correction_bias = nn.Parameter(
                torch.empty(config.n_routed_experts, dtype=torch.float32),
                requires_grad=False,
            )

        self.shared_experts = None
        if self.n_shared_experts:
            self.shared_experts = DeepseekV4MLP(
                hidden_size=config.hidden_size,
                intermediate_size=(
                    config.moe_intermediate_size * self.n_shared_experts
                ),
                hidden_act=config.hidden_act,
                swiglu_limit=self.swiglu_limit,
                quant_config=quant_config,
                prefix=f"{prefix}.shared_experts",
            )

        self.experts = FusedMoEFactory(
            # The Arm fused_cpp experts backend owns the combined routed+shared
            # execution.  Keeping shared experts outside MoERunner also lets the
            # upstream CPU backend compute them explicitly when fused_cpp is not
            # selected in non-strict mode.
            shared_experts=None,
            gate=self.gate,
            num_experts=config.n_routed_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            renormalize=config.norm_topk_prob,
            quant_config=quant_config,
            prefix=f"{prefix}.experts",
            scoring_func=self.scoring_func,
            routed_scaling_factor=self.routed_scaling_factor,
            e_score_correction_bias=self.gate.e_score_correction_bias,
            hash_indices_table=self.gate.tid2eid,
            swiglu_limit=self.swiglu_limit,
            router_logits_dtype=torch.float32,
            enable_eplb=False,
        )
        # The modular experts adapter needs the owning shared MLP during the
        # post-load routed+shared joint prepack.
        self.experts.routed_experts._dsv4_shared_experts = self.shared_experts

    def disable_placeholder_hash_routing(self) -> bool:
        """Use score routing when a checkpoint ships an empty hash table.

        Some DeepSeek V4 checkpoints reserve ``tid2eid`` for the hash-routing
        layers but fill the complete tensor with zero.  Treating that placeholder
        as a real lookup table sends every token to expert zero.  Keep the loaded
        parameter for checkpoint compatibility, while allowing the v0.28 router
        to use its normal sqrt-softplus score path.
        """
        table = typing.cast(torch.Tensor | None, self.gate.tid2eid)
        if table is None or torch.count_nonzero(table).item() != 0:
            return False
        router = typing.cast(typing.Any, self.experts.router)
        router._hash_indices_table = None
        logger.warning_once(
            "DeepSeek V4 layer %s has an all-zero tid2eid placeholder; "
            "using sqrt-softplus score routing",
            self.prefix,
        )
        return True

    def forward(
        self, hidden_states: torch.Tensor, input_ids: torch.Tensor | None = None
    ) -> torch.Tensor:
        with record_function_or_nullcontext("vllm::deepseek_v4_moe"):
            router = typing.cast(typing.Any, self.experts.router)
            if router._hash_indices_table is not None and input_ids is None:
                raise ValueError("DeepSeek V4 hash MoE routing requires input_ids")
            output = self.experts(
                hidden_states=hidden_states,
                router_logits=hidden_states,
                input_ids=input_ids,
            )
            kernel = getattr(
                self.experts.routed_experts.quant_method, "moe_kernel", None
            )
            fuses_shared = bool(
                kernel is not None
                and getattr(kernel.fused_experts, "fuses_shared_experts", False)
            )
            if self.shared_experts is not None and not fuses_shared:
                output = output + self.shared_experts(hidden_states)
            return output.reshape_as(hidden_states)

    def finalize_mega_moe_weights(self) -> None:
        return


class DeepseekV4DecoderLayer(nn.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str,
        topk_indices_buffer: torch.Tensor | None = None,
        **_: object,
    ) -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.hidden_size = config.hidden_size
        self.hc_mult = config.hc_mult
        self.hc_sinkhorn_iters = config.hc_sinkhorn_iters
        self.hc_eps = config.hc_eps
        self.hc_post_alpha = 2.0
        self.rms_norm_eps = config.rms_norm_eps

        self.attn = DeepseekV4CPUAttention(
            vllm_config,
            prefix=f"{prefix}.attn",
            topk_indices_buffer=topk_indices_buffer,
        )
        self.ffn = DeepseekV4MoE(vllm_config, prefix=f"{prefix}.ffn")
        self.attn_norm = RMSNorm(self.hidden_size, self.rms_norm_eps)
        self.ffn_norm = RMSNorm(self.hidden_size, self.rms_norm_eps)

        mix_hc = (2 + self.hc_mult) * self.hc_mult
        hc_dim = self.hc_mult * self.hidden_size
        self.hc_attn_fn = nn.Parameter(
            torch.empty(mix_hc, hc_dim, dtype=torch.float32), requires_grad=False
        )
        self.hc_ffn_fn = nn.Parameter(
            torch.empty(mix_hc, hc_dim, dtype=torch.float32), requires_grad=False
        )
        self.hc_attn_base = nn.Parameter(
            torch.empty(mix_hc, dtype=torch.float32), requires_grad=False
        )
        self.hc_ffn_base = nn.Parameter(
            torch.empty(mix_hc, dtype=torch.float32), requires_grad=False
        )
        self.hc_attn_scale = nn.Parameter(
            torch.empty(3, dtype=torch.float32), requires_grad=False
        )
        self.hc_ffn_scale = nn.Parameter(
            torch.empty(3, dtype=torch.float32), requires_grad=False
        )
        self._prepared_hc_attn_fn = None
        self._prepared_hc_ffn_fn = None

    def process_mhc_weights_after_loading(self) -> None:
        required = envs.VLLM_CPU_FUSED_CPP_STRICT
        if self._prepared_hc_attn_fn is None:
            self._prepared_hc_attn_fn = prepare_fused_cpp_mhc_weight(
                self.hc_attn_fn.detach(),
                kind="pre",
                required=required,
            )
        if self._prepared_hc_ffn_fn is None:
            self._prepared_hc_ffn_fn = prepare_fused_cpp_mhc_weight(
                self.hc_ffn_fn.detach(),
                kind="pre",
                required=required,
            )

    def _pre(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None,
        post_mix: torch.Tensor | None,
        res_mix: torch.Tensor | None,
        fn: torch.Tensor,
        scale: torch.Tensor,
        base: torch.Tensor,
        norm: RMSNorm,
        prepared_fn: typing.Any | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = broadcast_residual(x, self.hc_mult)
            if prepared_fn is not None:
                post_mix, res_mix, x = fused_cpp_mhc_pre_rmsnorm(
                    residual,
                    prepared_fn,
                    scale,
                    base,
                    norm.weight,
                    self.rms_norm_eps,
                    self.hc_eps,
                    self.hc_post_alpha,
                    self.hc_sinkhorn_iters,
                )
            else:
                post_mix, res_mix, x = mhc_pre(
                    residual,
                    fn,
                    scale,
                    base,
                    self.rms_norm_eps,
                    self.hc_eps,
                    self.hc_post_alpha,
                    self.hc_sinkhorn_iters,
                )
                x = norm(x)
            return residual, post_mix, res_mix, x
        assert post_mix is not None and res_mix is not None
        if prepared_fn is not None:
            return fused_cpp_mhc_post_pre_rmsnorm(
                x,
                residual,
                post_mix,
                res_mix,
                prepared_fn,
                scale,
                base,
                norm.weight,
                self.rms_norm_eps,
                self.hc_eps,
                self.hc_post_alpha,
                self.hc_sinkhorn_iters,
            )
        residual, post_mix, res_mix, x = mhc_fused_post_pre(
            x,
            residual,
            post_mix,
            res_mix,
            fn,
            scale,
            base,
            self.rms_norm_eps,
            self.hc_eps,
            self.hc_post_alpha,
            self.hc_sinkhorn_iters,
        )
        return residual, post_mix, res_mix, norm(x)

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        input_ids: torch.Tensor | None,
        post_mix: torch.Tensor | None = None,
        res_mix: torch.Tensor | None = None,
        residual: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        residual, post_mix, res_mix, x = self._pre(
            x,
            residual,
            post_mix,
            res_mix,
            self.hc_attn_fn,
            self.hc_attn_scale,
            self.hc_attn_base,
            self.attn_norm,
            self._prepared_hc_attn_fn,
        )
        x = self.attn(positions, x)
        residual, post_mix, res_mix, x = self._pre(
            x,
            residual,
            post_mix,
            res_mix,
            self.hc_ffn_fn,
            self.hc_ffn_scale,
            self.hc_ffn_base,
            self.ffn_norm,
            self._prepared_hc_ffn_fn,
        )
        x = self.ffn(x, input_ids)
        return x, residual, post_mix, res_mix


class DeepseekV4Model(nn.Module, EagleModelMixin):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.quant_config = quant_config
        self.hc_eps = config.hc_eps
        self.hc_mult = config.hc_mult
        self.hc_dim = self.hc_mult * config.hidden_size
        self.rms_norm_eps = config.rms_norm_eps
        self.topk_indices_buffer = torch.empty(
            vllm_config.scheduler_config.max_num_batched_tokens,
            config.index_topk,
            dtype=torch.int32,
        )

        if get_pp_group().is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=f"{prefix}.embed_tokens",
            )
        else:
            self.embed_tokens = PPMissingLayer()
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: DeepseekV4DecoderLayer(
                vllm_config,
                prefix=prefix,
                topk_indices_buffer=self.topk_indices_buffer,
            ),
            prefix=f"{prefix}.layers",
        )
        self.norm = (
            RMSNorm(config.hidden_size, self.rms_norm_eps)
            if get_pp_group().is_last_rank
            else PPMissingLayer()
        )
        self.hc_head_fn = nn.Parameter(
            torch.empty(self.hc_mult, self.hc_dim, dtype=torch.float32),
            requires_grad=False,
        )
        self.hc_head_base = nn.Parameter(
            torch.empty(self.hc_mult, dtype=torch.float32), requires_grad=False
        )
        self.hc_head_scale = nn.Parameter(
            torch.empty(1, dtype=torch.float32), requires_grad=False
        )
        self._prepared_hc_head_fn = None
        spec_config = vllm_config.speculative_config
        needs_mtp = spec_config is not None and (
            spec_config.use_eagle() or spec_config.uses_draft_model()
        )
        self._mtp_hidden_buffer = (
            torch.empty(
                vllm_config.scheduler_config.max_num_batched_tokens,
                self.hc_dim,
                dtype=vllm_config.model_config.dtype,
            )
            if get_pp_group().is_last_rank and needs_mtp
            else None
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def make_empty_intermediate_tensors(
        self, batch_size: int, dtype: torch.dtype, device: torch.device
    ) -> IntermediateTensors:
        return IntermediateTensors(
            {
                "hidden_states": torch.zeros(
                    batch_size,
                    self.hc_mult,
                    self.config.hidden_size,
                    dtype=dtype,
                    device=device,
                )
            }
        )

    def process_mhc_weights_after_loading(self) -> None:
        for layer in self.layers:
            if isinstance(layer, DeepseekV4DecoderLayer):
                layer.process_mhc_weights_after_loading()
        if get_pp_group().is_last_rank and self._prepared_hc_head_fn is None:
            self._prepared_hc_head_fn = prepare_fused_cpp_mhc_weight(
                self.hc_head_fn.detach(),
                kind="head",
                required=envs.VLLM_CPU_FUSED_CPP_STRICT,
            )
            if self._prepared_hc_head_fn is not None:
                logger.info_once(
                    "Using fused_cpp SVE DeepSeek V4 mHC pre, post-pre, "
                    "and post-head RMSNorm stages."
                )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        if get_pp_group().is_first_rank:
            hidden_states = (
                inputs_embeds
                if inputs_embeds is not None
                else self.embed_input_ids(input_ids)
            )
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]

        residual = post_mix = res_mix = None
        aux_hidden_states: list[torch.Tensor] = []
        layer = None
        for idx, layer in enumerate(
            islice(self.layers, self.start_layer, self.end_layer),
            start=self.start_layer,
        ):
            hidden_states, residual, post_mix, res_mix = layer(
                hidden_states,
                positions,
                input_ids,
                post_mix,
                res_mix,
                residual,
            )
            if idx + 1 in self.aux_hidden_state_layers:
                aux = mhc_post(hidden_states, residual, post_mix, res_mix).mean(dim=1)
                aux_hidden_states.append(aux)
        final_residual = None
        if layer is not None:
            assert residual is not None and post_mix is not None and res_mix is not None
            if get_pp_group().is_last_rank and self._prepared_hc_head_fn is not None:
                hidden_states, final_residual = fused_cpp_mhc_post_head_rmsnorm(
                    hidden_states,
                    residual,
                    post_mix,
                    res_mix,
                    self._prepared_hc_head_fn,
                    self.hc_head_scale,
                    self.hc_head_base,
                    self.norm.weight,
                    self.rms_norm_eps,
                    self.hc_eps,
                )
            else:
                hidden_states = mhc_post(
                    hidden_states,
                    residual,
                    post_mix,
                    res_mix,
                )

        if not get_pp_group().is_last_rank:
            return IntermediateTensors({"hidden_states": hidden_states})
        if self._mtp_hidden_buffer is not None:
            count = hidden_states.shape[0]
            mtp_hidden = final_residual if final_residual is not None else hidden_states
            self._mtp_hidden_buffer[:count].copy_(mtp_hidden.flatten(1))
        if final_residual is None:
            hidden_states = hc_head(
                hidden_states,
                self.hc_head_fn,
                self.hc_head_scale,
                self.hc_head_base,
                self.rms_norm_eps,
                self.hc_eps,
            )
            hidden_states = self.norm(hidden_states)
        return (
            (hidden_states, aux_hidden_states) if aux_hidden_states else hidden_states
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            ("gate_up_proj", "w1", 0),
            ("gate_up_proj", "w3", 1),
            ("attn.fused_wqa_wkv", "attn.wq_a", 0),
            ("attn.fused_wqa_wkv", "attn.wkv", 1),
            ("compressor.fused_wkv_wgate", "compressor.wkv", 0),
            ("compressor.fused_wkv_wgate", "compressor.wgate", 1),
        ]
        params = dict(self.named_parameters())
        loaded: set[str] = set()
        tp_size = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()
        local_heads = self.config.num_attention_heads // tp_size
        head_start = local_heads * tp_rank
        head_end = head_start + local_heads
        expert_mapping = self.get_expert_mapping()
        pad_shared = getattr(self.quant_config, "weight_block_size", None) is not None

        for name, loaded_weight in weights:
            if pad_shared and ".shared_experts." in name:
                loaded_weight = self._pad_shared_expert_weight(
                    self.quant_config, name, loaded_weight
                )
            for parameter_name, checkpoint_name, shard in stacked_params_mapping:
                if ".experts." in name or checkpoint_name not in name:
                    continue
                mapped = name.replace(checkpoint_name, parameter_name)
                if is_pp_missing_parameter(mapped, self):
                    break
                parameter = params[mapped]
                parameter.weight_loader(parameter, loaded_weight, shard)
                loaded.add(mapped)
                break
            else:
                if ".experts." in name:
                    if (
                        "weight_scale" in name
                        and loaded_weight.dtype == torch.float8_e8m0fnu
                    ):
                        loaded_weight = loaded_weight.view(torch.uint8)
                    mapped = name
                    for (
                        parameter_name,
                        checkpoint_name,
                        expert_id,
                        expert_shard,
                    ) in expert_mapping:
                        if checkpoint_name not in name:
                            continue
                        candidate = name.replace(checkpoint_name, parameter_name)
                        if is_pp_missing_parameter(candidate, self):
                            continue
                        parameter = params[candidate]
                        weight_loader = typing.cast(
                            Callable[..., bool], parameter.weight_loader
                        )
                        if weight_loader(
                            parameter,
                            loaded_weight,
                            candidate,
                            shard_id=expert_shard,
                            expert_id=expert_id,
                            return_success=True,
                        ):
                            mapped = candidate
                            break
                    loaded.add(mapped)
                    continue
                if "attn_sink" in name:
                    if is_pp_missing_parameter(name, self):
                        continue
                    local = loaded_weight[head_start:head_end]
                    params[name][: local.shape[0]].copy_(local)
                    loaded.add(name)
                    continue
                if is_pp_missing_parameter(name, self):
                    continue
                parameter = params[name]
                weight_loader = typing.cast(
                    Callable[..., typing.Any],
                    getattr(parameter, "weight_loader", default_weight_loader),
                )
                weight_loader(parameter, loaded_weight)
                loaded.add(name)
        return loaded

    @staticmethod
    def _pad_shared_expert_weight(
        quant_config: QuantizationConfig | None,
        name: str,
        loaded_weight: torch.Tensor,
    ) -> torch.Tensor:
        block_size = getattr(quant_config, "weight_block_size", None)
        assert block_size is not None
        step = 1 if name.endswith("weight_scale_inv") else block_size[0]
        dim = 1 if ".down_proj." in name else 0
        multiple = get_tensor_model_parallel_world_size() * step
        pad = (
            cdiv(loaded_weight.shape[dim], multiple) * multiple
            - loaded_weight.shape[dim]
        )
        if pad == 0:
            return loaded_weight
        shape = list(loaded_weight.shape)
        shape[dim] = pad
        return torch.cat((loaded_weight, loaded_weight.new_zeros(shape)), dim=dim)

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        return fused_moe_make_expert_params_mapping(
            self,
            ckpt_gate_proj_name="w1",
            ckpt_down_proj_name="w2",
            ckpt_up_proj_name="w3",
            num_experts=self.config.n_routed_experts,
        )


def _make_deepseek_v4_weights_mapper(expert_dtype: str) -> WeightsMapper:
    scale_regex = (
        {
            re.compile(r"(\.experts\.\d+\.w[123])\.scale$"): r"\1.weight_scale",
            re.compile(r"\.scale$"): ".weight_scale_inv",
        }
        if expert_dtype == "fp4"
        else {re.compile(r"\.scale$"): ".weight_scale_inv"}
    )
    return WeightsMapper(
        orig_to_new_prefix={
            "layers.": "model.layers.",
            "embed.": "model.embed.",
            "norm.": "model.norm.",
            "hc_head": "model.hc_head",
            "mtp.": "model.mtp.",
        },
        orig_to_new_regex=scale_regex,
        orig_to_new_suffix={
            "head.weight": "lm_head.weight",
            "embed.weight": "embed_tokens.weight",
            ".ffn.gate.bias": ".ffn.gate.e_score_correction_bias",
        },
        orig_to_new_substr={
            ".shared_experts.w2": ".shared_experts.down_proj",
        },
    )


class DeepseekV4MixtureOfExperts(MixtureOfExperts):
    moe_mlp_layers: list[DeepseekV4MoE]

    def extract_moe_parameters(self, example: DeepseekV4MoE | None) -> None:
        if example is None:
            self.num_moe_layers = self.num_expert_groups = 0
            self.num_logical_experts = self.num_physical_experts = 0
            self.num_local_physical_experts = self.num_routed_experts = 0
            self.num_shared_experts = self.num_redundant_experts = 0
            return
        self.num_logical_experts = example.n_logical_experts
        self.num_physical_experts = example.n_physical_experts
        self.num_local_physical_experts = example.n_local_physical_experts
        self.num_routed_experts = example.n_routed_experts
        self.num_shared_experts = example.n_shared_experts
        self.num_redundant_experts = example.n_redundant_experts

    def update_physical_experts_metadata(
        self, num_physical_experts: int, num_local_physical_experts: int
    ) -> None:
        if num_physical_experts != self.num_physical_experts:
            raise NotImplementedError("CPU fused_cpp MoE does not support EPLB")
        assert num_local_physical_experts == self.num_local_physical_experts


class DeepseekV4ForCausalLM(
    nn.Module, SupportsPP, SupportsEagle3, DeepseekV4MixtureOfExperts
):
    model_cls = DeepseekV4Model
    hf_to_vllm_mapper = _make_deepseek_v4_weights_mapper("fp4")

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        self.config = vllm_config.model_config.hf_config
        expert_dtype = getattr(self.config, "expert_dtype", "fp4")
        if expert_dtype != "fp4":
            self.hf_to_vllm_mapper = _make_deepseek_v4_weights_mapper(expert_dtype)
        self.model = self.model_cls(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        self.lm_head = (
            ParallelLMHead(
                self.config.vocab_size,
                self.config.hidden_size,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
            if get_pp_group().is_last_rank
            else PPMissingLayer()
        )
        self.logits_processor = LogitsProcessor(self.config.vocab_size)
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )
        self.set_moe_parameters()

    def set_moe_parameters(self) -> None:
        self.num_expert_groups = getattr(self.config, "n_group", 1)
        self.moe_layers: list[nn.Module] = []
        self.moe_mlp_layers = []
        example = None
        for layer in self.model.layers:
            if isinstance(layer, DeepseekV4DecoderLayer):
                example = layer.ffn
                self.moe_mlp_layers.append(layer.ffn)
                self.moe_layers.append(layer.ffn.experts)
        self.num_moe_layers = len(self.moe_layers)
        self.extract_moe_parameters(example)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        return self.model(input_ids, positions, intermediate_tensors, inputs_embeds)

    def get_mtp_target_hidden_states(self) -> torch.Tensor | None:
        return self.model._mtp_hidden_buffer

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        if getattr(self.config, "cpu_fp8_to_int8", False):
            from .fp8_requant import convert_fp8_checkpoint_for_cpu_w8a8

            block_size = tuple(
                getattr(self.config, "cpu_fp8_source_block_size", (128, 128))
            )
            rows_per_chunk = int(
                getattr(self.config, "cpu_fp8_conversion_rows_per_chunk", 2048)
            )
            weights = convert_fp8_checkpoint_for_cpu_w8a8(
                weights,
                block_size=typing.cast(tuple[int, int], block_size),
                rows_per_chunk=rows_per_chunk,
                tp_rank=get_tensor_model_parallel_rank(),
                tp_size=get_tensor_model_parallel_world_size(),
            )
        loader = AutoWeightsLoader(self, skip_substrs=["mtp."])
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)

    def process_weights_after_loading(self) -> None:
        self.model.process_mhc_weights_after_loading()
        for module in self.modules():
            if isinstance(module, DeepseekV4CPUAttention):
                module.process_weights_after_loading()
            elif isinstance(module, DeepseekV4MoE):
                module.disable_placeholder_hash_routing()

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        return self.model.get_expert_mapping()


__all__ = [
    "DeepseekV4DecoderLayer",
    "DeepseekV4ForCausalLM",
    "DeepseekV4MLP",
    "DeepseekV4MoE",
    "DeepseekV4Model",
]
