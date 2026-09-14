# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Environment-controlled per-token MoE route capture for CPU debugging."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import torch

from vllm.config import VllmConfig
from vllm.distributed import get_tensor_model_parallel_rank
from vllm.logger import init_logger

logger = init_logger(__name__)

ROUTE_CAPTURE_DIR_ENV = "DSV4_CPU_MOE_ROUTE_CAPTURE_DIR"
ROUTE_CAPTURE_CONTROL_ENV = "DSV4_CPU_MOE_ROUTE_CAPTURE_CONTROL"


class FileRouteRecorder:
    """Accumulate one request's logical expert IDs and save a compressed NPZ."""

    def __init__(
        self,
        output_dir: Path,
        control_path: Path,
        *,
        num_layers: int,
        num_experts: int,
        experts_per_token: int,
        num_shared_experts: int,
    ) -> None:
        self.output_dir = output_dir
        self.control_path = control_path
        self.num_layers = num_layers
        self.num_experts = num_experts
        self.experts_per_token = experts_per_token
        self.shared_expert_ids = np.arange(
            num_experts,
            num_experts + num_shared_experts,
            dtype=np.int16,
        )
        self.capture_id: str | None = None
        self.completed_capture_id: str | None = None
        self.control: dict[str, Any] | None = None
        self.routes: np.ndarray | None = None
        self.seen_layers = np.zeros(num_layers, dtype=np.bool_)

    def _read_control(self) -> dict[str, Any] | None:
        try:
            with self.control_path.open(encoding="utf-8") as file:
                control = json.load(file)
        except FileNotFoundError:
            return None
        if not isinstance(control, dict):
            raise ValueError("MoE route capture control must be a JSON object")
        for field in ("capture_id", "output_file", "prompt_token_ids"):
            if field not in control:
                raise ValueError(f"MoE route capture control is missing {field!r}")
        return control

    def _start_capture(
        self,
        control: dict[str, Any],
        topk_ids: torch.Tensor,
    ) -> None:
        capture_id = str(control["capture_id"])
        output_file = str(control["output_file"])
        if Path(output_file).name != output_file or not output_file.endswith(".npz"):
            raise ValueError("route output_file must be a basename ending in .npz")
        prompt_token_ids = control["prompt_token_ids"]
        if not isinstance(prompt_token_ids, list):
            raise ValueError("prompt_token_ids must be a list")
        if len(prompt_token_ids) != topk_ids.shape[0]:
            raise ValueError(
                "route token count does not match prompt: "
                f"routes={topk_ids.shape[0]}, prompt={len(prompt_token_ids)}"
            )
        self.capture_id = capture_id
        self.control = control
        self.routes = np.full(
            (len(prompt_token_ids), self.num_layers, self.experts_per_token),
            -1,
            dtype=np.int16,
        )
        self.seen_layers.fill(False)

    def capture(self, layer_id: int, topk_ids: torch.Tensor) -> None:
        control = self._read_control()
        if control is None:
            return
        capture_id = str(control["capture_id"])
        if capture_id == self.completed_capture_id:
            return
        if capture_id != self.capture_id:
            if self.routes is not None and not self.seen_layers.all():
                missing = np.flatnonzero(~self.seen_layers).tolist()
                raise RuntimeError(
                    f"capture {self.capture_id!r} ended with missing layers {missing}"
                )
            self._start_capture(control, topk_ids)
        assert self.routes is not None

        if not 0 <= layer_id < self.num_layers:
            raise IndexError(
                f"route layer {layer_id} is outside {self.num_layers} layers"
            )
        if self.seen_layers[layer_id]:
            raise RuntimeError(
                f"capture {capture_id!r} received layer {layer_id} more than once"
            )
        if topk_ids.shape != (
            self.routes.shape[0],
            self.experts_per_token,
        ):
            raise ValueError(
                f"unexpected route tensor shape {tuple(topk_ids.shape)}, expected "
                f"{(self.routes.shape[0], self.experts_per_token)}"
            )

        routes = topk_ids.detach().to(device="cpu", dtype=torch.int16).numpy()
        if routes.size and (routes.min() < 0 or routes.max() >= self.num_experts):
            raise ValueError(
                f"expert IDs must be in [0, {self.num_experts}), got "
                f"[{routes.min()}, {routes.max()}]"
            )
        self.routes[:, layer_id, :] = routes
        self.seen_layers[layer_id] = True
        if self.seen_layers.all():
            self._flush()

    def _flush(self) -> None:
        assert self.control is not None
        assert self.capture_id is not None
        assert self.routes is not None
        self.output_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.output_dir / str(self.control["output_file"])
        temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
        metadata = {
            key: value
            for key, value in self.control.items()
            if key not in {"prompt_token_ids"}
        }
        with temporary_path.open("wb") as file:
            np.savez_compressed(
                file,
                expert_ids=self.routes,
                prompt_token_ids=np.asarray(
                    self.control["prompt_token_ids"], dtype=np.int32
                ),
                layer_ids=np.arange(self.num_layers, dtype=np.int16),
                shared_expert_ids=self.shared_expert_ids,
                metadata_json=np.asarray(
                    json.dumps(metadata, ensure_ascii=False, sort_keys=True)
                ),
            )
        os.replace(temporary_path, output_path)
        logger.info(
            "Saved MoE routes for capture %s to %s with shape %s",
            self.capture_id,
            output_path,
            self.routes.shape,
        )
        self.completed_capture_id = self.capture_id
        self.capture_id = None
        self.control = None
        self.routes = None
        self.seen_layers.fill(False)


def _hf_int(config: object, name: str, default: int = 0) -> int:
    value = getattr(config, name, default)
    return int(value or 0)


def maybe_bind_file_route_recorder(
    model: torch.nn.Module,
    vllm_config: VllmConfig,
) -> FileRouteRecorder | None:
    """Bind a rank-zero file recorder when the capture environment is set."""

    output_dir = os.environ.get(ROUTE_CAPTURE_DIR_ENV)
    if not output_dir:
        return None
    control_path = os.environ.get(ROUTE_CAPTURE_CONTROL_ENV)
    if not control_path:
        raise ValueError(
            f"{ROUTE_CAPTURE_CONTROL_ENV} is required when {ROUTE_CAPTURE_DIR_ENV} "
            "is set"
        )
    if get_tensor_model_parallel_rank() != 0:
        return None

    from vllm.model_executor.layers.fused_moe.router.base_router import BaseRouter
    from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner

    model_config = vllm_config.model_config
    hf_config = model_config.hf_text_config
    recorder = FileRouteRecorder(
        Path(output_dir),
        Path(control_path),
        num_layers=model_config.get_total_num_hidden_layers(),
        num_experts=model_config.get_num_experts(),
        experts_per_token=model_config.get_num_experts_per_tok(),
        num_shared_experts=_hf_int(hf_config, "n_shared_experts"),
    )
    bound_layers: set[int] = set()
    for module in model.modules():
        if not isinstance(module, MoERunner):
            continue
        if not isinstance(module.router, BaseRouter):
            raise ValueError(
                "file route capture requires BaseRouter, got "
                f"{type(module.router).__name__}"
            )
        if module.router.capture_fn is not None:
            raise ValueError(
                f"MoE layer {module.layer_id} already has a route capture callback"
            )
        module.router.set_capture_fn(partial(recorder.capture, module.layer_id))
        bound_layers.add(module.layer_id)

    expected_layers = set(range(recorder.num_layers))
    if bound_layers != expected_layers:
        raise ValueError(
            "file route capture did not bind every layer: "
            f"missing={sorted(expected_layers - bound_layers)}, "
            f"unexpected={sorted(bound_layers - expected_layers)}"
        )
    model._file_route_recorder = recorder
    logger.info(
        "Enabled file MoE route capture for %d layers at %s",
        recorder.num_layers,
        output_dir,
    )
    return recorder


def write_route_control(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically publish one request's capture metadata to worker processes."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, sort_keys=True)
        file.write("\n")
    os.replace(temporary_path, path)


__all__ = [
    "FileRouteRecorder",
    "ROUTE_CAPTURE_CONTROL_ENV",
    "ROUTE_CAPTURE_DIR_ENV",
    "maybe_bind_file_route_recorder",
    "write_route_control",
]
