# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU MTP draft model for DeepSeek V4."""

import typing
from collections.abc import Callable, Iterable

import regex as re
import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import (
    fused_moe_make_expert_params_mapping,
)
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
from vllm.model_executor.model_loader.mtp_validation import (
    is_mtp_completeness_check_enabled,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.deepseek_mtp import SharedHead
from vllm.model_executor.models.deepseek_v2 import get_spec_layer_idx_from_weight_name
from vllm.model_executor.models.utils import maybe_prefix
from vllm.sequence import IntermediateTensors

from .attention import DeepseekV4CPUAttention
from .mhc import hc_head, mhc_post
from .model import DeepseekV4DecoderLayer, DeepseekV4Model
from .ops import rms_norm

logger = init_logger(__name__)

_EXPERT_SCALE_RE = re.compile(r"\.experts\.\d+\.w[123]\.scale$")


class DeepSeekV4MultiTokenPredictorLayer(nn.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,
        topk_indices_buffer: torch.Tensor,
        prefix: str,
    ) -> None:
        super().__init__()
        assert vllm_config.speculative_config is not None
        config = vllm_config.speculative_config.draft_model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.rms_norm_eps = config.rms_norm_eps
        self.hc_eps = config.hc_eps
        self.hc_mult = config.hc_mult
        self.hc_dim = self.hc_mult * config.hidden_size

        self.enorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.hnorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.e_proj = ReplicatedLinear(
            config.hidden_size,
            config.hidden_size,
            bias=False,
            return_bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.e_proj",
        )
        self.h_proj = ReplicatedLinear(
            config.hidden_size,
            config.hidden_size,
            bias=False,
            return_bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.h_proj",
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
        self.shared_head = SharedHead(
            config=config, prefix=prefix, quant_config=quant_config
        )
        self.mtp_block = DeepseekV4DecoderLayer(
            vllm_config,
            prefix,
            topk_indices_buffer=topk_indices_buffer,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        previous_hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_index: int = 0,
    ) -> torch.Tensor:
        del input_ids, spec_step_index
        assert inputs_embeds is not None
        previous_hidden_states = previous_hidden_states.reshape(
            -1, self.hc_mult, self.config.hidden_size
        )
        inputs_embeds = torch.where(
            positions.unsqueeze(-1) == 0,
            torch.zeros_like(inputs_embeds),
            inputs_embeds,
        )
        inputs_embeds = rms_norm(
            inputs_embeds, self.enorm.weight, self.enorm.variance_epsilon
        )
        previous_hidden_states = rms_norm(
            previous_hidden_states,
            self.hnorm.weight,
            self.hnorm.variance_epsilon,
        )
        hidden_states = self.h_proj(previous_hidden_states) + self.e_proj(
            inputs_embeds
        ).unsqueeze(-2)
        hidden_states, residual, post_mix, res_mix = self.mtp_block(
            positions=positions,
            x=hidden_states,
            input_ids=None,
        )
        return mhc_post(hidden_states, residual, post_mix, res_mix).flatten(1)


class DeepSeekV4MultiTokenPredictor(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.mtp_start_layer_idx = config.num_hidden_layers
        self.num_mtp_layers = config.num_nextn_predict_layers
        self.topk_indices_buffer = torch.empty(
            vllm_config.scheduler_config.max_num_batched_tokens,
            config.index_topk,
            dtype=torch.int32,
        )
        self.layers = nn.ModuleDict(
            {
                str(index): DeepSeekV4MultiTokenPredictorLayer(
                    vllm_config,
                    self.topk_indices_buffer,
                    f"{prefix}.layers.{index}",
                )
                for index in range(
                    self.mtp_start_layer_idx,
                    self.mtp_start_layer_idx + self.num_mtp_layers,
                )
            }
        )
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            prefix=maybe_prefix(prefix, "embed_tokens"),
        )
        self.logits_processor = LogitsProcessor(config.vocab_size)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        previous_hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        step = spec_step_idx % self.num_mtp_layers
        return self.layers[str(self.mtp_start_layer_idx + step)](
            input_ids,
            positions,
            previous_hidden_states,
            inputs_embeds,
            step,
        )

    def compute_logits(
        self, hidden_states: torch.Tensor, spec_step_idx: int = 0
    ) -> torch.Tensor:
        step = spec_step_idx % self.num_mtp_layers
        layer = self.layers[str(self.mtp_start_layer_idx + step)]
        hidden_states = hidden_states.reshape(
            -1, layer.hc_mult, layer.config.hidden_size
        )
        hidden_states = hc_head(
            hidden_states,
            layer.hc_head_fn,
            layer.hc_head_scale,
            layer.hc_head_base,
            layer.rms_norm_eps,
            layer.hc_eps,
        )
        hidden_states = rms_norm(
            hidden_states,
            layer.shared_head.norm.weight,
            layer.shared_head.norm.variance_epsilon,
        )
        return self.logits_processor(layer.shared_head.head, hidden_states)


class DeepSeekV4MTP(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        self.config = vllm_config.model_config.hf_config
        self.quant_config = vllm_config.quant_config
        self.pad_shared_expert = (
            getattr(self.quant_config, "weight_block_size", None) is not None
        )
        self.model = DeepSeekV4MultiTokenPredictor(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "model"),
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        del intermediate_tensors
        assert input_ids is not None
        return self.model(
            input_ids,
            positions,
            hidden_states,
            inputs_embeds,
            spec_step_idx,
        )

    def compute_logits(
        self, hidden_states: torch.Tensor, spec_step_idx: int = 0
    ) -> torch.Tensor | None:
        return self.model.compute_logits(hidden_states, spec_step_idx)

    def _rewrite_spec_layer_name(self, spec_layer: int, name: str) -> str:
        top_level = (
            "embed_tokens",
            "enorm",
            "hnorm",
            "h_proj",
            "e_proj",
            "shared_head",
            "hc_head_fn",
            "hc_head_base",
            "hc_head_scale",
        )
        if not any(component in name for component in top_level):
            return name.replace(
                f"model.layers.{spec_layer}.",
                f"model.layers.{spec_layer}.mtp_block.",
            )
        if "embed_tokens" in name:
            return name.replace(f"model.layers.{spec_layer}.", "model.")
        return name

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        params = dict(self.named_parameters())
        loaded: set[str] = set()
        tp_size = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()
        local_heads = self.config.num_attention_heads // tp_size
        head_start = local_heads * tp_rank
        head_end = head_start + local_heads
        expert_mapping = fused_moe_make_expert_params_mapping(
            self,
            ckpt_gate_proj_name="w1",
            ckpt_down_proj_name="w2",
            ckpt_up_proj_name="w3",
            num_experts=self.config.n_routed_experts,
        )
        expert_scale_suffix = (
            ".weight_scale"
            if getattr(self.config, "expert_dtype", "fp4") == "fp4"
            else ".weight_scale_inv"
        )
        stacked = [
            ("gate_up_proj", "w1", 0),
            ("gate_up_proj", "w3", 1),
            ("attn.fused_wqa_wkv", "attn.wq_a", 0),
            ("attn.fused_wqa_wkv", "attn.wkv", 1),
            ("compressor.fused_wkv_wgate", "compressor.wkv", 0),
            ("compressor.fused_wkv_wgate", "compressor.wgate", 1),
        ]

        for original_name, loaded_weight in weights:
            parts = original_name.split(".")
            mtp_index = next((int(part) for part in parts if part.isdigit()), 0)
            name = original_name.replace(
                f"mtp.{mtp_index}.",
                f"model.layers.{self.config.num_hidden_layers + mtp_index}.",
            )
            spec_layer = get_spec_layer_idx_from_weight_name(self.config, name)
            if spec_layer is None:
                continue
            for old, new in {
                ".emb.tok_emb.weight": ".embed_tokens.weight",
                ".head.weight": ".shared_head.head.weight",
                ".norm.weight": ".shared_head.norm.weight",
            }.items():
                name = name.replace(old, new)
            name = self._rewrite_spec_layer_name(spec_layer, name)
            if name.endswith(".scale"):
                suffix = (
                    expert_scale_suffix
                    if _EXPERT_SCALE_RE.search(name)
                    else ".weight_scale_inv"
                )
                name = name.removesuffix(".scale") + suffix
            name = name.replace(".shared_experts.w2", ".shared_experts.down_proj")
            if self.pad_shared_expert and ".shared_experts." in name:
                loaded_weight = DeepseekV4Model._pad_shared_expert_weight(
                    self.quant_config, name, loaded_weight
                )

            handled = False
            for parameter_name, checkpoint_name, stacked_shard in stacked:
                if ".experts." in name or checkpoint_name not in name:
                    continue
                mapped = name.replace(checkpoint_name, parameter_name)
                parameter = params[mapped]
                parameter.weight_loader(parameter, loaded_weight, stacked_shard)
                loaded.add(mapped)
                handled = True
                break
            if handled:
                continue
            if ".experts." in name:
                if (
                    "weight_scale" in name
                    and loaded_weight.dtype == torch.float8_e8m0fnu
                ):
                    loaded_weight = loaded_weight.view(torch.uint8)
                for (
                    parameter_name,
                    checkpoint_name,
                    expert_id,
                    expert_shard,
                ) in expert_mapping:
                    if checkpoint_name not in name:
                        continue
                    mapped = name.replace(checkpoint_name, parameter_name)
                    loader = typing.cast(
                        Callable[..., bool], params[mapped].weight_loader
                    )
                    if loader(
                        params[mapped],
                        loaded_weight,
                        mapped,
                        shard_id=expert_shard,
                        expert_id=expert_id,
                        return_success=True,
                    ):
                        loaded.add(mapped)
                        break
                continue
            if "attn_sink" in name:
                narrow = loaded_weight[head_start:head_end]
                params[name][: narrow.shape[0]].copy_(narrow)
                loaded.add(name)
                continue
            name = name.replace(".ffn.gate.bias", ".ffn.gate.e_score_correction_bias")
            parameter = params[name]
            loader = typing.cast(
                Callable[..., typing.Any],
                getattr(parameter, "weight_loader", default_weight_loader),
            )
            loader(parameter, loaded_weight)
            loaded.add(name)

        found_layers = {
            layer
            for name in loaded
            if (layer := get_spec_layer_idx_from_weight_name(self.config, name))
            is not None
        }
        for layer in range(
            self.model.mtp_start_layer_idx,
            self.model.mtp_start_layer_idx + self.model.num_mtp_layers,
        ):
            if layer not in found_layers and is_mtp_completeness_check_enabled():
                raise ValueError(f"MTP layer {layer} weights are missing")
        logger.info_once("MTP draft model loaded: %d params", len(loaded))
        return loaded

    def process_weights_after_loading(self) -> None:
        for module in self.modules():
            if isinstance(module, DeepseekV4CPUAttention):
                module.process_weights_after_loading()


__all__ = ["DeepSeekV4MTP"]
