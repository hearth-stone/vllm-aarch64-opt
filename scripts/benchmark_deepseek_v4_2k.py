# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""随机排序串行请求 DeepSeek V4 服务，统计 2K 输入的 E2E 延迟。"""

import argparse
import json
import math
import random
import statistics
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

DEFAULT_DATASET = (
    Path(__file__).resolve().parents[1]
    / "benchmarks/datasets/deepseek_v4/zh2048-public-100.jsonl"
)


def select_cases(dataset: Path, count: int, seed: int) -> list[dict]:
    cases = [json.loads(line) for line in dataset.read_text().splitlines() if line]
    if not 1 <= count <= len(cases):
        raise ValueError(f"请求条数必须在 1–{len(cases)} 之间")
    if len({case["case_id"] for case in cases}) != len(cases):
        raise ValueError("数据集存在重复 case_id")
    for case in cases:
        if len(case["prompt_token_ids"]) != 2048:
            raise ValueError(f"{case['case_id']} 的输入不是 2048 token")
    random.Random(seed).shuffle(cases)
    return cases[:count]


def summarize(rows: list[dict]) -> dict:
    """只统计成功请求，百分位使用排序样本的线性插值。"""
    values = sorted(row["e2e_s"] for row in rows if row["success"])

    def percentile(q: float) -> float | None:
        if not values:
            return None
        position = (len(values) - 1) * q
        low, high = math.floor(position), math.ceil(position)
        return values[low] + (values[high] - values[low]) * (position - low)

    return {
        "attempted": len(rows),
        "successful": len(values),
        "failed": len(rows) - len(values),
        "avg_s": statistics.mean(values) if values else None,
        "p50_s": percentile(0.50),
        "p90_s": percentile(0.90),
        "p95_s": percentile(0.95),
        "p99_s": percentile(0.99),
        "min_s": min(values) if values else None,
        "max_s": max(values) if values else None,
    }


def send_request(case: dict, base_url: str, model: str, timeout: float) -> dict:
    payload = {
        "model": model,
        "prompt": case["prompt_token_ids"],
        "max_tokens": 1,
        "temperature": 0,
        "ignore_eos": True,
        "stream": False,
    }
    request = Request(
        f"{base_url.rstrip('/')}/v1/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    row = {
        "case_id": case["case_id"],
        "title": case["title"],
        "request": payload,
        "success": False,
    }
    row["start_unix"] = time.time()
    begin = time.perf_counter()
    try:
        with urlopen(request, timeout=timeout) as response:
            body = response.read()
        # 计时覆盖 HTTP 请求及完整响应读取；解析 JSON 不计入服务请求延迟。
        row["e2e_s"] = time.perf_counter() - begin
        row["end_unix"] = time.time()
        result = json.loads(body)
        row["response"] = result
        if result["usage"]["prompt_tokens"] != 2048:
            raise ValueError("服务返回的输入 token 数不是 2048")
        if result["usage"]["completion_tokens"] != 1:
            raise ValueError("服务返回的输出 token 数不是 1")
        row["text"] = result["choices"][0]["text"]
        row["success"] = True
    except HTTPError as exc:
        row["error"] = f"HTTP {exc.code}: {exc.read().decode(errors='replace')}"
    except (URLError, TimeoutError, OSError, ValueError, KeyError, IndexError) as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
    if "e2e_s" not in row:
        row["e2e_s"] = time.perf_counter() - begin
        row["end_unix"] = time.time()
    return row


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def analyze_results(run: dict) -> dict:
    summary = summarize(run["requests"])
    summary.update(
        requested=run["protocol"]["requests"],
        warmup_attempted=len(run["warmups"]),
        warmup_failed=sum(not row["success"] for row in run["warmups"]),
        complete=not run["interrupted"]
        and summary["successful"] == run["protocol"]["requests"],
        interrupted=run["interrupted"],
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analyze", type=Path, help="读取已有 requests.json 重算统计")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--base-url", default="http://127.0.0.1:8018")
    parser.add_argument("--model", default="deepseek-v4")
    parser.add_argument("--requests", type=int, default=100, help="不重复的测量条数")
    parser.add_argument("--interval-seconds", type=float, default=5)
    parser.add_argument("--seed", type=int, default=0, help="固定种子可复现请求顺序")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--timeout-seconds", type=float, default=240)
    parser.add_argument("--output-dir", type=Path, help="新建结果目录，不覆盖已有目录")
    args = parser.parse_args(argv)
    if args.analyze:
        try:
            summary = analyze_results(json.loads(args.analyze.read_text()))
            if args.output_dir:
                args.output_dir.mkdir(parents=True, exist_ok=False)
                write_json(args.output_dir / "summary.json", summary)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            parser.error(f"无法分析 {args.analyze}: {exc}")
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    if args.output_dir is None:
        parser.error("请求模式需要 --output-dir")
    if (
        args.warmup < 0
        or not math.isfinite(args.interval_seconds)
        or args.interval_seconds < 0
        or not math.isfinite(args.timeout_seconds)
        or args.timeout_seconds <= 0
    ):
        parser.error("预热次数和间隔必须非负，超时必须为正的有限数值")
    try:
        cases = select_cases(args.dataset, args.requests, args.seed)
        args.output_dir.mkdir(parents=True, exist_ok=False)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    protocol = {
        "dataset": str(args.dataset.resolve()),
        "base_url": args.base_url,
        "model": args.model,
        "requests": args.requests,
        "interval_seconds": args.interval_seconds,
        "warmup": args.warmup,
        "seed": args.seed,
        "timeout_seconds": args.timeout_seconds,
        "input_tokens": 2048,
        "output_tokens": 1,
        "case_order": [case["case_id"] for case in cases],
    }
    run = {"protocol": protocol, "requests": [], "warmups": [], "interrupted": False}
    result_file = args.output_dir / "requests.json"
    write_json(result_file, run)
    try:
        total = args.warmup + len(cases)
        for index in range(total):
            is_warmup = index < args.warmup
            case = cases[0] if is_warmup else cases[index - args.warmup]
            row = send_request(case, args.base_url, args.model, args.timeout_seconds)
            row["index"] = index if is_warmup else index - args.warmup
            run["warmups" if is_warmup else "requests"].append(row)
            write_json(result_file, run)
            write_json(args.output_dir / "summary.json", analyze_results(run))
            label = "warmup" if is_warmup else f"{len(run['requests'])}/{len(cases)}"
            print(
                f"{label} {row['case_id']}: {row['e2e_s']:.4f}s "
                f"{'OK' if row['success'] else row['error']}",
                flush=True,
            )
            if not row["success"]:
                break
            # 间隔从响应结束后开始，串行发送，不在最后一条之后等待。
            if index + 1 < total and args.interval_seconds:
                time.sleep(args.interval_seconds)
    except KeyboardInterrupt:
        run["interrupted"] = True
    write_json(result_file, run)
    summary = analyze_results(run)
    write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0 if summary["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
