#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measure sequential 2048-token requests against an already running CPU API."""

import argparse
import json
import math
import statistics
import time
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen


def complete(base_url: str, model: str, case: dict, max_tokens: int) -> dict:
    payload = {
        "model": model,
        "prompt": case["prompt_token_ids"],
        "max_tokens": max_tokens,
        "temperature": 0,
        "ignore_eos": True,
        "stream": False,
    }
    request = Request(
        f"{base_url.rstrip('/')}/v1/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    started = time.perf_counter()
    try:
        with urlopen(request, timeout=3600) as response:
            result = json.load(response)
    except HTTPError as exc:
        raise RuntimeError(f"HTTP {exc.code}: {exc.read().decode()}") from exc
    elapsed = time.perf_counter() - started
    usage = result["usage"]
    if usage["prompt_tokens"] != 2048:
        raise ValueError(f"Expected 2048 prompt tokens, got {usage}")
    return {
        "case_id": case["case_id"],
        "elapsed_s": elapsed,
        "prompt_tokens": usage["prompt_tokens"],
        "completion_tokens": usage["completion_tokens"],
        "input_tokens_per_s": usage["prompt_tokens"] / elapsed,
        "response": result,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8004")
    parser.add_argument(
        "--model", required=True, help="Served model ID from /v1/models"
    )
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--requests", type=int, default=20)
    parser.add_argument("--interval-seconds", type=float, default=10)
    parser.add_argument("--max-tokens", type=int, default=1)
    args = parser.parse_args()
    if args.requests < 1 or args.max_tokens < 1 or args.interval_seconds < 0:
        parser.error(
            "requests/max-tokens must be positive; interval must be nonnegative"
        )
    cases = [
        json.loads(line) for line in args.cases.read_text().splitlines() if line.strip()
    ]
    if len(cases) < args.requests + 1:
        parser.error("cases must contain one warmup plus the requested measured cases")
    for case in cases[: args.requests + 1]:
        if len(case["prompt_token_ids"]) != 2048:
            parser.error(f"{case['case_id']}: expected 2048 prompt tokens")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = vars(args) | {"cases": str(args.cases), "output_dir": str(args.output_dir)}
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    print("Warmup (excluded from summary)", flush=True)
    warmup = complete(args.base_url, args.model, cases[0], args.max_tokens)
    (args.output_dir / "warmup.json").write_text(json.dumps(warmup) + "\n")

    rows = []
    with (args.output_dir / "requests.jsonl").open("w") as output:
        for index, case in enumerate(cases[1 : args.requests + 1], 1):
            time.sleep(args.interval_seconds)
            row = complete(args.base_url, args.model, case, args.max_tokens)
            row["request_index"] = index
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
            output.flush()
            rows.append(row)
            print(
                f"{index}/{args.requests} {row['case_id']}: "
                f"{row['elapsed_s']:.3f}s, "
                f"{row['input_tokens_per_s']:.1f} input tokens/s",
                flush=True,
            )
    latencies = sorted(row["elapsed_s"] for row in rows)
    summary = {
        "requests": len(rows),
        "mean_elapsed_s": statistics.mean(latencies),
        "median_elapsed_s": statistics.median(latencies),
        "p95_elapsed_s": latencies[math.ceil(0.95 * len(rows)) - 1],
        "input_tokens_per_s": sum(row["prompt_tokens"] for row in rows)
        / sum(latencies),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
