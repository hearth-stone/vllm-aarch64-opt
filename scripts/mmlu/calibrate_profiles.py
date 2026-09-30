#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Generate rank-local fused_cpp MoE profiles for a CPU tensor-parallel service."""

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--first-cpu", type=int, required=True)
    parser.add_argument("--rank-count", type=int, required=True)
    parser.add_argument("--threads-per-rank", type=int, required=True)
    args = parser.parse_args()
    if args.first_cpu < 0 or args.rank_count < 1 or args.threads_per_rank < 1:
        parser.error("CPU IDs must be non-negative; ranks and threads must be positive")

    from fused_cpp import moe

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for rank in range(args.rank_count):
        first = args.first_cpu + rank * args.threads_per_rank
        cpu_ids = tuple(range(first, first + args.threads_per_rank))
        path = args.output_dir / f"rank{rank}.json"
        result = moe.calibrate_moe_planner_quick(cpu_ids, output=path)
        print(f"rank={rank} cpus={first}-{cpu_ids[-1]} profile={result.output_path}")


if __name__ == "__main__":
    main()
