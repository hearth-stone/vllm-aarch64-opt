#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Create a no-copy DeepSeek V4 FP8-to-CPU-W8A8 model view."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from vllm.models.deepseek_v4.cpu.fp8_requant import (
    make_cpu_w8a8_quantization_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows-per-chunk", type=int, default=2048)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = args.source.resolve(strict=True)
    output = args.output.resolve()
    if source == output:
        raise ValueError("source and output directories must differ")
    if args.rows_per_chunk <= 0:
        raise ValueError("--rows-per-chunk must be positive")

    with (source / "config.json").open(encoding="utf-8") as handle:
        config = json.load(handle)
    if config.get("model_type") != "deepseek_v4":
        raise ValueError("source model must have model_type=deepseek_v4")
    source_quant = config.get("quantization_config") or {}
    if source_quant.get("quant_method") != "fp8":
        raise ValueError("source model must use a block-FP8 checkpoint")
    block_size = source_quant.get("weight_block_size")
    if block_size != [128, 128]:
        raise ValueError(f"expected FP8 block size [128, 128], got {block_size}")

    output.mkdir(parents=True, exist_ok=True)
    for child in source.iterdir():
        if child.name == "config.json":
            continue
        destination = output / child.name
        if destination.is_symlink() and destination.resolve() == child.resolve():
            continue
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(f"refusing to replace {destination}")
        destination.symlink_to(child)

    config["cpu_fp8_to_int8"] = True
    config["cpu_fp8_source_block_size"] = block_size
    config["cpu_fp8_conversion_rows_per_chunk"] = args.rows_per_chunk
    config["quantization_config"] = make_cpu_w8a8_quantization_config(
        int(config["num_hidden_layers"])
    )
    temporary = output / "config.json.tmp"
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    os.replace(temporary, output / "config.json")
    print(output)


if __name__ == "__main__":
    main()
