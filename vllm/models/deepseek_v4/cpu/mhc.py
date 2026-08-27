# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Torch reference implementation of DeepSeek V4 multi-head compression.

Keep this module self-contained: importing the generic MHC package registers
CUDA, Triton and TileLang kernels at module import time, which is undesirable
for the CPU-isolated DeepSeek V4 path.
"""

import typing
from functools import lru_cache

import torch


@lru_cache(maxsize=1)
def _load_fused_cpp_mhc() -> typing.Any | None:
    try:
        import fused_cpp.deepseek_v4_mhc as fused_mhc
    except (ImportError, AttributeError):
        return None
    required = (
        "prepare_mhc_weight",
        "mhc_pre_rmsnorm_sve_candidate",
        "mhc_post_pre_rmsnorm_sve_candidate",
        "mhc_post_hc_head_rmsnorm_sve_candidate",
    )
    if not fused_mhc._HAS_DEEPSEEK_V4_MHC_SVE_PROJECTION:
        return None
    if any(not hasattr(fused_mhc, name) for name in required):
        return None
    return fused_mhc


def prepare_fused_cpp_mhc_weight(
    fn: torch.Tensor,
    *,
    kind: str,
    required: bool,
) -> typing.Any | None:
    fused_mhc = _load_fused_cpp_mhc()
    if fused_mhc is None:
        if required:
            raise RuntimeError("required fused_cpp DeepSeek V4 mHC is unavailable")
        return None
    return fused_mhc.prepare_mhc_weight(fn, kind=kind)


def fused_cpp_mhc_pre_rmsnorm(
    residual: torch.Tensor,
    prepared_fn: typing.Any,
    scale: torch.Tensor,
    base: torch.Tensor,
    norm_weight: torch.Tensor,
    rms_eps: float,
    hc_eps: float,
    post_alpha: float,
    sinkhorn_iters: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    fused_mhc = _load_fused_cpp_mhc()
    if fused_mhc is None:
        raise RuntimeError("fused_cpp DeepSeek V4 mHC disappeared after preparation")
    return fused_mhc.mhc_pre_rmsnorm_sve_candidate(
        residual,
        prepared_fn,
        scale,
        base,
        norm_weight,
        rms_eps=rms_eps,
        hc_pre_eps=hc_eps,
        hc_sinkhorn_eps=hc_eps,
        hc_post_mult_value=post_alpha,
        sinkhorn_repeat=sinkhorn_iters,
        norm_eps=rms_eps,
        num_threads=torch.get_num_threads(),
    )


def fused_cpp_mhc_post_pre_rmsnorm(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_mix: torch.Tensor,
    res_mix: torch.Tensor,
    prepared_fn: typing.Any,
    scale: torch.Tensor,
    base: torch.Tensor,
    norm_weight: torch.Tensor,
    rms_eps: float,
    hc_eps: float,
    post_alpha: float,
    sinkhorn_iters: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    fused_mhc = _load_fused_cpp_mhc()
    if fused_mhc is None:
        raise RuntimeError("fused_cpp DeepSeek V4 mHC disappeared after preparation")
    return fused_mhc.mhc_post_pre_rmsnorm_sve_candidate(
        x,
        residual,
        post_mix,
        res_mix,
        prepared_fn,
        scale,
        base,
        norm_weight,
        rms_eps=rms_eps,
        hc_pre_eps=hc_eps,
        hc_sinkhorn_eps=hc_eps,
        hc_post_mult_value=post_alpha,
        sinkhorn_repeat=sinkhorn_iters,
        norm_eps=rms_eps,
        num_threads=torch.get_num_threads(),
    )


def fused_cpp_mhc_post_head_rmsnorm(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_mix: torch.Tensor,
    res_mix: torch.Tensor,
    prepared_fn: typing.Any,
    scale: torch.Tensor,
    base: torch.Tensor,
    norm_weight: torch.Tensor,
    rms_eps: float,
    hc_eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    fused_mhc = _load_fused_cpp_mhc()
    if fused_mhc is None:
        raise RuntimeError("fused_cpp DeepSeek V4 mHC disappeared after preparation")
    return fused_mhc.mhc_post_hc_head_rmsnorm_sve_candidate(
        x,
        residual,
        post_mix,
        res_mix,
        prepared_fn,
        scale,
        base,
        norm_weight,
        rms_eps=rms_eps,
        hc_eps=hc_eps,
        norm_eps=rms_eps,
        num_threads=torch.get_num_threads(),
    )


def mhc_pre(
    residual: torch.Tensor,
    fn: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    rms_eps: float,
    hc_eps: float,
    post_alpha: float,
    sinkhorn_iters: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    hc_mult, hidden_size = residual.shape[-2:]
    outer_shape = residual.shape[:-2]
    flat = residual.reshape(-1, hc_mult, hidden_size)
    x = flat.reshape(flat.shape[0], -1).float()

    mixes = x @ fn.t()
    sqrsum = x.square().sum(dim=-1, keepdim=True)
    mixes *= torch.rsqrt(sqrsum / (hc_mult * hidden_size) + rms_eps)

    pre_logits = mixes[:, :hc_mult] * scale[0] + base[:hc_mult]
    pre_mix = torch.sigmoid(pre_logits) + hc_eps
    post_logits = (
        mixes[:, hc_mult : 2 * hc_mult] * scale[1] + base[hc_mult : 2 * hc_mult]
    )
    post_mix = torch.sigmoid(post_logits) * post_alpha

    comb_logits = mixes[:, 2 * hc_mult :].reshape(-1, hc_mult, hc_mult) * scale[
        2
    ] + base[2 * hc_mult :].reshape(1, hc_mult, hc_mult)
    comb_mix = torch.softmax(comb_logits, dim=-1) + hc_eps
    comb_mix /= comb_mix.sum(dim=-2, keepdim=True) + hc_eps
    for _ in range(max(0, sinkhorn_iters - 1)):
        comb_mix /= comb_mix.sum(dim=-1, keepdim=True) + hc_eps
        comb_mix /= comb_mix.sum(dim=-2, keepdim=True) + hc_eps

    layer_input = (pre_mix.unsqueeze(-1) * flat.float()).sum(dim=1)
    return (
        post_mix.reshape(*outer_shape, hc_mult, 1),
        comb_mix.reshape(*outer_shape, hc_mult, hc_mult),
        layer_input.to(residual.dtype).reshape(*outer_shape, hidden_size),
    )


def mhc_post(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_mix: torch.Tensor,
    res_mix: torch.Tensor,
) -> torch.Tensor:
    mixed_residual = torch.einsum(
        "...ij,...ih->...jh", res_mix.float(), residual.float()
    )
    post_term = post_mix.float() * x.unsqueeze(-2).float()
    return (mixed_residual + post_term).to(residual.dtype)


def mhc_fused_post_pre(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_mix: torch.Tensor,
    res_mix: torch.Tensor,
    fn: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    rms_eps: float,
    hc_eps: float,
    post_alpha: float,
    sinkhorn_iters: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    residual = mhc_post(x, residual, post_mix, res_mix)
    post_mix, res_mix, x = mhc_pre(
        residual,
        fn,
        scale,
        base,
        rms_eps,
        hc_eps,
        post_alpha,
        sinkhorn_iters,
    )
    return residual, post_mix, res_mix, x


def hc_head(
    residual: torch.Tensor,
    fn: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    rms_eps: float,
    hc_eps: float,
) -> torch.Tensor:
    """Collapse ``[..., hc_mult, hidden]`` residual streams to one stream."""

    hc_mult, hidden_size = residual.shape[-2:]
    outer_shape = residual.shape[:-2]
    flat = residual.reshape(-1, hc_mult, hidden_size)
    x = flat.reshape(flat.shape[0], -1).float()
    mixes = x @ fn.t()
    sqrsum = x.square().sum(dim=-1, keepdim=True)
    mixes *= torch.rsqrt(sqrsum / (hc_mult * hidden_size) + rms_eps)
    gates = torch.sigmoid(mixes * scale.reshape(1, 1) + base) + hc_eps
    output = (gates.unsqueeze(-1) * flat.float()).sum(dim=1)
    return output.to(residual.dtype).reshape(*outer_shape, hidden_size)


def broadcast_residual(x: torch.Tensor, hc_mult: int) -> torch.Tensor:
    """Create the training-time replicated HC streams for the first layer."""

    if x.ndim != 2:
        return x
    return x.unsqueeze(-2).expand(*x.shape[:-1], hc_mult, x.shape[-1]).contiguous()


__all__ = [
    "broadcast_residual",
    "fused_cpp_mhc_post_head_rmsnorm",
    "fused_cpp_mhc_post_pre_rmsnorm",
    "fused_cpp_mhc_pre_rmsnorm",
    "hc_head",
    "mhc_fused_post_pre",
    "mhc_post",
    "mhc_pre",
    "prepare_fused_cpp_mhc_weight",
]
