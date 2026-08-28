# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Streaming FP8 checkpoint conversion for DeepSeek V4 CPU W8A8."""

import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

_LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.")
_EXPERT_RE = re.compile(
    r"(?:^|\.)ffn\.(?:experts\.\d+|shared_experts)\.w[123]\.weight$"
)
_EXPERT_COLUMN_RE = re.compile(
    r"(?:^|\.)ffn\.(?:experts\.\d+|shared_experts)\.w[13]\.weight$"
)
_ATTENTION_INT8_SUFFIXES = (
    ".attn.wq_b.weight",
    ".attn.wo_b.weight",
    ".attn.indexer.wq_b.weight",
)
_COLUMN_PARALLEL_SUFFIXES = (
    ".attn.wq_b.weight",
    ".attn.wo_a.weight",
)
_TP_PRE_SHARDED_ATTR = "_vllm_tp_pre_sharded"
_FP8_DTYPES = tuple(
    dtype
    for dtype in (
        getattr(torch, "float8_e4m3fn", None),
        getattr(torch, "float8_e4m3fnuz", None),
    )
    if dtype is not None
)


@dataclass
class _PendingWeight:
    name: str
    value: torch.Tensor


def is_cpu_w8a8_target(name: str) -> bool:
    """Return whether a checkpoint weight uses an existing CPU W8A8 kernel."""
    return bool(_EXPERT_RE.search(name)) or name.endswith(_ATTENTION_INT8_SUFFIXES)


def is_column_parallel_fp8_weight(name: str) -> bool:
    """Return whether a V4 FP8 checkpoint matrix is TP-sharded over rows."""
    return bool(_EXPERT_COLUMN_RE.search(name)) or name.endswith(
        _COLUMN_PARALLEL_SUFFIXES
    )


def _slice_column_parallel_pair(
    name: str,
    weight: torch.Tensor,
    scales: torch.Tensor,
    *,
    tp_rank: int,
    tp_size: int,
    block_rows: int,
) -> tuple[torch.Tensor, torch.Tensor, bool]:
    if tp_size == 1 or not is_column_parallel_fp8_weight(name):
        return weight, scales, False
    if not 0 <= tp_rank < tp_size:
        raise ValueError(f"invalid TP rank {tp_rank} for TP size {tp_size}")
    if weight.shape[0] % tp_size != 0:
        raise ValueError(
            f"column-parallel rows {weight.shape[0]} are not divisible by TP {tp_size}"
        )
    rows_per_rank = weight.shape[0] // tp_size
    row_start = tp_rank * rows_per_rank
    if row_start % block_rows or rows_per_rank % block_rows:
        raise ValueError(
            "column-parallel FP8 shard boundaries must align to scale blocks: "
            f"start={row_start}, rows={rows_per_rank}, block_rows={block_rows}"
        )
    scale_start = row_start // block_rows
    scale_rows = rows_per_rank // block_rows
    return (
        weight.narrow(0, row_start, rows_per_rank),
        scales.narrow(0, scale_start, scale_rows),
        True,
    )


def _expanded_block_scales(
    scales: torch.Tensor,
    *,
    row_start: int,
    row_end: int,
    columns: int,
    block_size: tuple[int, int],
) -> torch.Tensor:
    block_rows, block_columns = block_size
    first_block = row_start // block_rows
    last_block = (row_end + block_rows - 1) // block_rows
    expanded = scales[first_block:last_block].float().repeat_interleave(
        block_rows, dim=0
    )
    offset = row_start - first_block * block_rows
    expanded = expanded[offset : offset + row_end - row_start]
    return expanded.repeat_interleave(block_columns, dim=1)[:, :columns]


def convert_block_fp8_weight(
    weight: torch.Tensor,
    scales: torch.Tensor,
    *,
    to_int8: bool,
    block_size: tuple[int, int] = (128, 128),
    rows_per_chunk: int = 2048,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Convert one block-scaled FP8 matrix without a full FP32 materialization.

    Args:
        weight: FP8 matrix in ``[N, K]`` checkpoint layout.
        scales: Dequantization scales in ``[ceil(N/Bn), ceil(K/Bk)]``.
        to_int8: Produce per-output-channel symmetric INT8 when true, otherwise
            BF16.
        block_size: FP8 scale block shape ``(Bn, Bk)``.
        rows_per_chunk: Maximum output rows expanded to FP32 at once.

    Returns:
        Converted weight and an FP32 ``[N, 1]`` scale for INT8, or ``None``
        for BF16.

    Raises:
        ValueError: If tensor dtypes, ranks, scale shape, or chunk size do not
            satisfy the conversion contract.
    """
    if weight.dtype not in _FP8_DTYPES or weight.ndim != 2:
        raise ValueError("DeepSeek V4 conversion requires a 2D E4M3 FP8 weight")
    if scales.dtype != torch.float32 or scales.ndim != 2:
        raise ValueError("DeepSeek V4 conversion requires a 2D FP32 block scale")
    if rows_per_chunk <= 0:
        raise ValueError("rows_per_chunk must be positive")

    rows, columns = weight.shape
    block_rows, block_columns = block_size
    expected = (
        (rows + block_rows - 1) // block_rows,
        (columns + block_columns - 1) // block_columns,
    )
    if tuple(scales.shape) != expected:
        raise ValueError(
            f"FP8 scale shape mismatch: expected {expected}, got {tuple(scales.shape)}"
        )

    chunk_rows = max(block_rows, rows_per_chunk // block_rows * block_rows)
    output_dtype = torch.int8 if to_int8 else torch.bfloat16
    output = torch.empty(weight.shape, dtype=output_dtype, device=weight.device)
    output_scales = (
        torch.empty((rows, 1), dtype=torch.float32, device=weight.device)
        if to_int8
        else None
    )
    vmax = torch.iinfo(torch.int8).max

    for row_start in range(0, rows, chunk_rows):
        row_end = min(row_start + chunk_rows, rows)
        expanded_scales = _expanded_block_scales(
            scales,
            row_start=row_start,
            row_end=row_end,
            columns=columns,
            block_size=block_size,
        )
        dequantized = weight[row_start:row_end].float() * expanded_scales
        if output_scales is None:
            output[row_start:row_end].copy_(dequantized.to(torch.bfloat16))
            continue
        channel_scales = dequantized.abs().amax(dim=1, keepdim=True) / vmax
        safe_scales = torch.where(
            channel_scales > 0,
            channel_scales,
            torch.ones_like(channel_scales),
        )
        quantized = dequantized.div(safe_scales).round().clamp(-vmax, vmax)
        output[row_start:row_end].copy_(quantized.to(torch.int8))
        output_scales[row_start:row_end].copy_(safe_scales)

    return output, output_scales


def convert_fp8_checkpoint_for_cpu_w8a8(
    weights: Iterable[tuple[str, torch.Tensor]],
    *,
    block_size: tuple[int, int] = (128, 128),
    rows_per_chunk: int = 2048,
    tp_rank: int = 0,
    tp_size: int = 1,
) -> Iterator[tuple[str, torch.Tensor]]:
    """Convert a DeepSeek V4 block-FP8 stream to the CPU hybrid format.

    FP8 weights backed by an existing CPU W8A8 kernel become INT8 plus an
    FP32 per-channel scale. Other FP8 weights become BF16. The input stream is
    expected to keep each ``.scale`` close to its corresponding ``.weight``;
    only unmatched pairs remain buffered.

    Args:
        weights: Raw checkpoint ``(name, tensor)`` pairs before name mapping.
        block_size: FP8 checkpoint scale block shape.
        rows_per_chunk: Maximum rows converted through FP32 at once.
        tp_rank: Tensor-parallel rank receiving this stream.
        tp_size: Tensor-parallel world size. Column-parallel matrices are
            narrowed as mmap views before conversion.

    Yields:
        Converted checkpoint tensors suitable for compressed-tensors INT8
        modules and unquantized BF16 modules.

    Raises:
        ValueError: If an FP8 weight or block scale is missing its pair.
    """
    pending_scales: dict[str, torch.Tensor] = {}
    pending_weights: dict[str, _PendingWeight] = {}
    converted_int8 = 0
    converted_bf16 = 0
    layer_counts: dict[int, list[int]] = {}

    def convert_pair(
        stem: str, pending: _PendingWeight, scale: torch.Tensor
    ) -> Iterator[tuple[str, torch.Tensor]]:
        nonlocal converted_int8, converted_bf16
        source_weight, source_scale, pre_sharded = _slice_column_parallel_pair(
            pending.name,
            pending.value,
            scale,
            tp_rank=tp_rank,
            tp_size=tp_size,
            block_rows=block_size[0],
        )
        to_int8 = is_cpu_w8a8_target(pending.name)
        converted, int8_scale = convert_block_fp8_weight(
            source_weight,
            source_scale,
            to_int8=to_int8,
            block_size=block_size,
            rows_per_chunk=rows_per_chunk,
        )
        layer_match = _LAYER_RE.search(pending.name)
        if layer_match is not None:
            counts = layer_counts.setdefault(int(layer_match.group(1)), [0, 0])
            counts[0 if to_int8 else 1] += 1
        if to_int8:
            converted_int8 += 1
            assert int8_scale is not None
            if pre_sharded:
                setattr(converted, _TP_PRE_SHARDED_ATTR, True)
                setattr(int8_scale, _TP_PRE_SHARDED_ATTR, True)
            yield pending.name, converted
            yield f"{stem}.weight_scale", int8_scale
        else:
            converted_bf16 += 1
            if pre_sharded:
                setattr(converted, _TP_PRE_SHARDED_ATTR, True)
            yield pending.name, converted

    for name, value in weights:
        if name.endswith(".scale"):
            stem = name.removesuffix(".scale")
            pending = pending_weights.pop(stem, None)
            if pending is None:
                pending_scales[stem] = value
            else:
                yield from convert_pair(stem, pending, value)
            continue

        if name.endswith(".weight") and value.dtype in _FP8_DTYPES:
            stem = name.removesuffix(".weight")
            scale = pending_scales.pop(stem, None)
            pending = _PendingWeight(name, value)
            if scale is None:
                pending_weights[stem] = pending
            else:
                yield from convert_pair(stem, pending, scale)
            continue

        yield name, value

    if pending_weights or pending_scales:
        missing_scales = sorted(pending_weights)
        missing_weights = sorted(pending_scales)
        raise ValueError(
            "unpaired DeepSeek V4 FP8 checkpoint tensors: "
            f"missing_scales={missing_scales[:8]}, "
            f"missing_weights={missing_weights[:8]}"
        )
    for layer, (int8_count, bf16_count) in sorted(layer_counts.items()):
        logger.info(
            "Converted DeepSeek V4 layer %d FP8 weights: INT8=%d, BF16=%d",
            layer,
            int8_count,
            bf16_count,
        )
    logger.info(
        "Converted DeepSeek V4 FP8 checkpoint weights: INT8=%d, BF16=%d, "
        "TP=%d/%d, threads=%d",
        converted_int8,
        converted_bf16,
        tp_rank,
        tp_size,
        torch.get_num_threads(),
    )


def make_cpu_w8a8_quantization_config(num_hidden_layers: int) -> dict[str, object]:
    """Build the compressed-tensors W8A8 config for a DeepSeek V4 CPU view."""
    ignore = ["head"]
    for layer in range(num_hidden_layers):
        prefix = f"layers.{layer}.attn"
        ignore.extend((f"{prefix}.wq_a", f"{prefix}.wkv", f"{prefix}.wo_a"))
        ignore.extend((f"{prefix}.compressor.wgate", f"{prefix}.compressor.wkv"))
        if layer >= 2 and layer % 2 == 0:
            ignore.extend(
                (
                    f"{prefix}.indexer.weights_proj",
                    f"{prefix}.indexer.compressor.wgate",
                    f"{prefix}.indexer.compressor.wkv",
                )
            )
    ignore.extend(
        (
            "mtp.0.attn.wq_a",
            "mtp.0.attn.wkv",
            "mtp.0.attn.wo_a",
            "mtp.0.attn.wq_b",
            "mtp.0.attn.wo_b",
            "mtp.0.head",
        )
    )
    return {
        "config_groups": {
            "group_0": {
                "targets": ["Linear"],
                "input_activations": {
                    "dynamic": True,
                    "num_bits": 8,
                    "observer": "memoryless",
                    "strategy": "token",
                    "symmetric": True,
                    "type": "int",
                },
                "weights": {
                    "dynamic": False,
                    "num_bits": 8,
                    "observer": "minmax",
                    "strategy": "channel",
                    "symmetric": True,
                    "type": "int",
                },
            }
        },
        "format": "int-quantized",
        "ignore": ignore,
        "quant_method": "compressed-tensors",
        "quantization_status": "compressed",
    }
