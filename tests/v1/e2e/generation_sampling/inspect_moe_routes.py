#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Inspect per-token MoE expert IDs captured by generation_prefill_suite.py."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--route-dir", required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--token-position", type=int)
    parser.add_argument("--output")
    return parser.parse_args()


def load_manifest(route_dir: Path, case_id: str) -> dict[str, object]:
    matches = []
    with (route_dir / "route_manifest.jsonl").open(encoding="utf-8") as file:
        for line in file:
            row = json.loads(line)
            if str(row["case_id"]) == case_id:
                matches.append(row)
    if not matches:
        raise ValueError(f"case {case_id!r} is not present in the route manifest")
    measured = [row for row in matches if row["phase"] == "measured"]
    return measured[0] if measured else matches[0]


def main() -> None:
    args = parse_args()
    route_dir = Path(args.route_dir)
    manifest = load_manifest(route_dir, args.case_id)
    route_path = route_dir / str(manifest["route_file"])
    with np.load(route_path, allow_pickle=False) as payload:
        expert_ids = payload["expert_ids"]
        prompt_token_ids = payload["prompt_token_ids"]
        shared_expert_ids = payload["shared_expert_ids"].tolist()
    if not 0 <= args.layer < expert_ids.shape[1]:
        raise ValueError(f"layer must be in [0, {expert_ids.shape[1]})")
    positions = (
        [args.token_position]
        if args.token_position is not None
        else range(expert_ids.shape[0])
    )
    output_path = Path(args.output) if args.output else None
    output_file = output_path.open("w", encoding="utf-8") if output_path else None
    try:
        for position in positions:
            if not 0 <= position < expert_ids.shape[0]:
                raise ValueError(
                    f"token position must be in [0, {expert_ids.shape[0]})"
                )
            row = {
                "case_id": args.case_id,
                "layer": args.layer,
                "token_position": position,
                "token_id": int(prompt_token_ids[position]),
                "routed_expert_ids": expert_ids[position, args.layer].tolist(),
                "shared_expert_ids": shared_expert_ids,
            }
            text = json.dumps(row, ensure_ascii=False, sort_keys=True)
            if output_file is None:
                print(text)
            else:
                output_file.write(text + "\n")
    finally:
        if output_file is not None:
            output_file.close()


if __name__ == "__main__":
    main()
