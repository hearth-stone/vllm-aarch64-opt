#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measure fixed-length Chinese requests against a ready TP4 service."""

import argparse
import json
import math
import re
import statistics
import time
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen


def snapshot(ranks: dict[str, str]) -> dict:
    result = {}
    for rank, pid in ranks.items():
        raw = Path(f"/proc/{pid}/stat").read_text()
        fields = raw[raw.rfind(")") + 2 :].split()
        result[rank] = {
            "minor_faults": int(fields[7]),
            "major_faults": int(fields[9]),
            "rss_bytes": int(fields[21]) * 4096,
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="deepseek-v4")
    parser.add_argument("--service-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--requests", type=int, default=60)
    parser.add_argument("--interval-seconds", type=float, default=5)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--input-tokens", type=int, default=2048)
    parser.add_argument("--case-id", default="zh2048-001")
    parser.add_argument(
        "--distinct-cases",
        action="store_true",
        help="Measure the first N cases instead of repeating --case-id",
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8018")
    parser.add_argument(
        "--cases",
        type=Path,
        default=Path(
            "/mnt/models/.cache/zhangxu/benchmarks/"
            "fused-attn80-continuous30-20260821/cases31.jsonl"
        ),
    )
    args = parser.parse_args()
    if args.requests < 1 or args.warmup < 0 or args.interval_seconds < 0:
        parser.error("requests must be positive; warmup and interval nonnegative")
    cases = [
        json.loads(line) for line in args.cases.read_text().splitlines() if line.strip()
    ]
    if args.distinct_cases:
        if len(cases) < args.requests:
            parser.error("not enough distinct cases for the requested count")
        selected = cases[: args.requests]
    else:
        case = next(row for row in cases if row["case_id"] == args.case_id)
        selected = [case] * args.requests
    if any(len(case["prompt_token_ids"]) != args.input_tokens for case in selected):
        parser.error(f"selected cases must contain {args.input_tokens} prompt tokens")
    model = args.model
    with urlopen(f"{args.base_url}/v1/models", timeout=10) as response:
        models = json.load(response)
    if model not in {item["id"] for item in models["data"]}:
        parser.error(f"{model} is not served at {args.base_url}")
    ranks = dict(
        re.findall(
            r"Worker_TP(\d) pid=(\d+)",
            (args.service_run / "server.log").read_text(),
        )
    )
    if len(ranks) != 4:
        parser.error("service log must identify four TP workers")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(
        {
            "case_ids": [case["case_id"] for case in selected],
            "input_tokens": args.input_tokens,
            "output_tokens": 1,
            "worker_ranks": ranks,
        }
    )
    (args.output_dir / "protocol.json").write_text(json.dumps(config, indent=2))
    payload = {
        "model": model,
        "prompt": selected[0]["prompt_token_ids"],
        "max_tokens": 1,
        "temperature": 0,
        "ignore_eos": True,
        "stream": False,
        "logprobs": 5,
    }
    rows = []
    warmups = []
    for index in range(args.warmup + args.requests):
        case = selected[0] if index < args.warmup else selected[index - args.warmup]
        payload["prompt"] = case["prompt_token_ids"]
        before = snapshot(ranks)
        request = Request(
            f"{args.base_url}/v1/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        start_unix = time.time()
        begin = time.perf_counter()
        try:
            with urlopen(request, timeout=240) as response:
                result = json.load(response)
        except HTTPError as exc:
            (args.output_dir / "error.txt").write_text(exc.read().decode())
            raise
        elapsed = time.perf_counter() - begin
        end_unix = time.time()
        after = snapshot(ranks)
        if result["usage"]["prompt_tokens"] != args.input_tokens:
            raise ValueError(f"Unexpected input usage: {result['usage']}")
        if result["usage"]["completion_tokens"] != 1:
            raise ValueError(f"Unexpected output usage: {result['usage']}")
        row = {
            "index": index - args.warmup,
            "case_id": case["case_id"],
            "elapsed_s": elapsed,
            "start_unix": start_unix,
            "end_unix": end_unix,
            "response": result,
            "after": after,
            "faults": {
                rank: {
                    key: after[rank][key] - before[rank][key]
                    for key in ("minor_faults", "major_faults")
                }
                for rank in ranks
            },
        }
        is_warmup = index < args.warmup
        target = warmups if is_warmup else rows
        target.append(row)
        name = "warmup.json" if is_warmup else "requests.json"
        temporary = args.output_dir / f"{name}.tmp"
        temporary.write_text(json.dumps(target, indent=2))
        temporary.replace(args.output_dir / name)
        label = "warmup" if is_warmup else f"{len(rows)}/{args.requests}"
        print(f"{label}: {elapsed:.4f}s", flush=True)
        if args.interval_seconds and index < args.warmup + args.requests - 1:
            time.sleep(args.interval_seconds)
    latencies = sorted(row["elapsed_s"] for row in rows)
    position = 0.95 * (len(latencies) - 1)
    low, high = math.floor(position), math.ceil(position)
    summary = {
        "requests": len(rows),
        "mean_s": statistics.mean(latencies),
        "median_s": statistics.median(latencies),
        "min_s": min(latencies),
        "max_s": max(latencies),
        "p95_s": latencies[low] + (latencies[high] - latencies[low]) * (position - low),
        "over_3s": sum(value > 3 for value in latencies),
        "over_5s": sum(value > 5 for value in latencies),
        "minor_faults_by_rank": {
            rank: sum(r["faults"][rank]["minor_faults"] for r in rows) for rank in ranks
        },
        "major_faults_by_rank": {
            rank: sum(r["faults"][rank]["major_faults"] for r in rows) for rank in ranks
        },
        "rss_delta_MiB_by_rank": {
            rank: (
                rows[-1]["after"][rank]["rss_bytes"]
                - rows[0]["after"][rank]["rss_bytes"]
            )
            / 2**20
            for rank in ranks
        },
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
