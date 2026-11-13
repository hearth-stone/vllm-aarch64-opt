#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare complete Chinese answers at single/batch load and sustained concurrency."""

import argparse
import concurrent.futures
import importlib.util
import json
import math
import re
import statistics
import threading
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def save(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def prepare(args: argparse.Namespace) -> None:
    from tokenizers import Tokenizer

    encoding_file = Path(__file__).resolve().parents[1] / (
        "vllm/tokenizers/deepseek_v4_encoding.py"
    )
    spec = importlib.util.spec_from_file_location("dsv4_encoding", encoding_file)
    assert spec is not None and spec.loader is not None
    encoding = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(encoding)
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    source = [
        json.loads(line)
        for line in (args.dataset / "cases_8192.jsonl").read_text().splitlines()
    ]
    rows = []
    for length, count in ((512, 4), (1024, 4), (2048, 8), (4096, 4), (8192, 4)):
        for _ in range(count):
            index = len(rows)
            case_id = f"验收{index + 1:03d}"
            expected = {
                "编号": case_id,
                "机构": ["青岚", "白石", "松川", "星河"][index % 4] + "市文化局",
                "日期": f"2026年{index % 9 + 1}月{index % 27 + 1}日",
                "人数": str(31 + index * 7),
            }
            record = (
                "【测试记录】"
                + "；".join(f"{key}：{value}" for key, value in expected.items())
                + "。【测试记录结束】"
            )
            content = (
                "下面是新闻背景材料，其中插入了一条专门用于验收的虚构测试记录。"
                "请只从【测试记录】提取编号、机构、日期、人数，所有值用字符串。"
                "只输出这四个键的JSON对象，不解释，不输出背景中的其他事实。\n"
                "__BACKGROUND_BEFORE__" + record + "__BACKGROUND_AFTER__"
                "\n请按要求回答。"
            )
            rendered = encoding.encode_messages(
                [{"role": "user", "content": content}], thinking_mode="chat"
            )
            first, rest = rendered.split("__BACKGROUND_BEFORE__")
            middle, last = rest.split("__BACKGROUND_AFTER__")
            pieces = [
                tokenizer.encode(p, add_special_tokens=False).ids
                for p in (first, middle, last)
            ]
            budget = length - sum(len(p) for p in pieces)
            position = (0.1, 0.5, 0.9)[index % 3]
            cut = int(budget * position)
            background = source[index]["prompt_token_ids"][:budget]
            # Move split boundaries away from incomplete UTF-8 byte tokens.
            while "\ufffd" in tokenizer.decode(background[:cut]):
                cut -= 1
            while "\ufffd" in tokenizer.decode(background[cut:]):
                background.pop()
            background.extend(
                [tokenizer.encode("\n").ids[0]] * (budget - len(background))
            )
            prompt_ids = pieces[0] + background[:cut] + pieces[1]
            prompt_ids += background[cut:] + pieces[2]
            assert len(prompt_ids) == length
            prompt = tokenizer.decode(prompt_ids, skip_special_tokens=False)
            assert "\ufffd" not in prompt and record in prompt
            rows.append(
                {
                    "case_id": case_id,
                    "input_tokens": length,
                    "prompt_token_ids": prompt_ids,
                    "prompt": prompt,
                    "expected": expected,
                    "record_position_fraction": position,
                    "source_title": source[index]["source_title"],
                    "source_url": source[index]["source_url"],
                    "source_license": source[index]["source_license"],
                }
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    )
    print(f"Prepared {len(rows)} exact-length controlled extraction cases", flush=True)


def judge(text: str, expected: dict, finish_reason: str | None) -> dict:
    stripped = text.strip()
    if stripped.startswith("```json\n") and stripped.endswith("```"):
        stripped = stripped[8:-3].strip()
    elif stripped.startswith("```\n") and stripped.endswith("```"):
        stripped = stripped[4:-3].strip()
    try:
        parsed = json.loads(stripped)
        parse_error = None
    except json.JSONDecodeError as exc:
        parsed, parse_error = None, str(exc)
    garbled = "\ufffd" in text or any(
        ord(char) < 32 and char not in "\n\r\t" for char in text
    )
    return {
        "parsed": parsed,
        "parse_error": parse_error,
        "garbled": garbled,
        "empty": not text.strip(),
        "truncated": finish_reason == "length",
        "correct": parsed == expected and not garbled and finish_reason == "stop",
    }


def complete(
    base_url: str,
    model: str,
    case: dict,
    max_tokens: int,
    timeout: float,
    barrier: threading.Barrier | None = None,
) -> dict:
    payload = {
        "model": model,
        "prompt": case["prompt_token_ids"],
        "temperature": 0,
        "seed": 0,
        "max_tokens": max_tokens,
        "stream": True,
        "logprobs": 1,
        "stream_options": {"include_usage": True},
    }
    request = Request(
        f"{base_url}/v1/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    if barrier is not None:
        barrier.wait(timeout=30)
    begin = time.perf_counter()
    row = {
        "case_id": case["case_id"],
        "input_tokens": case["input_tokens"],
        "start_unix": time.time(),
        "expected": case["expected"],
    }
    text, chunks, usage, token_times = [], [], None, []
    finish_reason = None
    done = False
    try:
        with urlopen(request, timeout=timeout) as response:
            for raw in response:
                line = raw.decode("utf-8").strip()
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    done = True
                    break
                event = json.loads(data)
                elapsed = time.perf_counter() - begin
                chunks.append({"elapsed_s": elapsed, "event": event})
                if "error" in event:
                    raise ValueError(f"stream error: {event['error']}")
                if event.get("usage") is not None:
                    usage = event["usage"]
                for choice in event.get("choices", []):
                    text.append(choice.get("text", ""))
                    probs = choice.get("logprobs") or {}
                    logs = probs.get("token_logprobs") or []
                    if not all(v is not None and math.isfinite(v) for v in logs):
                        raise ValueError("Non-finite/missing token logprob")
                    token_times.extend([elapsed] * len(logs))
                    if choice.get("finish_reason") is not None:
                        finish_reason = choice["finish_reason"]
        if not done or usage is None or finish_reason is None:
            raise ValueError("Incomplete SSE response or missing final usage")
        if usage["prompt_tokens"] != case["input_tokens"]:
            raise ValueError(f"Input token count differs: {usage}")
        if usage["completion_tokens"] < 1:
            raise ValueError(f"Empty token output: {usage}")
        row.update(
            usage=usage,
            finish_reason=finish_reason,
            judgment=judge("".join(text), case["expected"], finish_reason),
        )
        row["error"] = None
    except (HTTPError, URLError, TimeoutError, OSError, ValueError) as exc:
        detail = (
            exc.read().decode("utf-8", errors="replace")
            if isinstance(exc, HTTPError)
            else str(exc)
        )
        row["error"] = f"{type(exc).__name__}: {detail}"
        row["judgment"] = {
            "correct": False,
            "garbled": False,
            "empty": not text,
            "truncated": False,
        }
    row.update(
        text="".join(text),
        chunks=chunks,
        elapsed_s=time.perf_counter() - begin,
        end_unix=time.time(),
        ttft_s=token_times[0] if token_times else None,
        tpot_s=(
            (token_times[-1] - token_times[0]) / (len(token_times) - 1)
            if len(token_times) > 1
            else None
        ),
    )
    return row


def percentiles(values: list[float]) -> dict:
    if not values:
        return {}
    ordered = sorted(values)
    result = {
        "mean": statistics.mean(values),
        "stddev": statistics.pstdev(values),
        "min": min(values),
        "max": max(values),
    }
    for percentile in (50, 90, 95, 99):
        position = percentile / 100 * (len(ordered) - 1)
        low, high = math.floor(position), math.ceil(position)
        result[f"p{percentile}"] = ordered[low] + (ordered[high] - ordered[low]) * (
            position - low
        )
    return result


def summarize(rows: list[dict], duration: float) -> dict:
    return {
        "requests": len(rows),
        "wall_s": duration,
        "correct": sum(r["judgment"]["correct"] for r in rows),
        "errors": sum(r["error"] is not None for r in rows),
        "garbled": sum(r["judgment"]["garbled"] for r in rows),
        "empty": sum(r["judgment"]["empty"] for r in rows),
        "truncated": sum(r["judgment"]["truncated"] for r in rows),
        "request_per_s": len(rows) / duration,
        "output_token_per_s": sum(
            r.get("usage", {}).get("completion_tokens", 0) for r in rows
        )
        / duration,
        "e2e_s": percentiles([r["elapsed_s"] for r in rows]),
        "ttft_s": percentiles([r["ttft_s"] for r in rows if r["ttft_s"] is not None]),
        "tpot_s": percentiles([r["tpot_s"] for r in rows if r["tpot_s"] is not None]),
    }


def validate_api_batch(response: dict, cases: list[dict]) -> list[dict]:
    choices = sorted(response["choices"], key=lambda choice: choice["index"])
    if [choice["index"] for choice in choices] != list(range(len(cases))):
        raise ValueError("Batch response has missing or duplicate choice indices")
    if response["usage"]["prompt_tokens"] != sum(c["input_tokens"] for c in cases):
        raise ValueError("Batch input token count differs")
    rows = []
    for case, choice in zip(cases, choices):
        logs = (choice.get("logprobs") or {}).get("token_logprobs") or []
        if not logs or not all(v is not None and math.isfinite(v) for v in logs):
            raise ValueError("Missing or non-finite batch token logprobs")
        rows.append(
            {
                "case_id": case["case_id"],
                "input_tokens": case["input_tokens"],
                "text": choice["text"],
                "finish_reason": choice["finish_reason"],
                "completion_tokens": len(logs),
                "choice_index": choice["index"],
                "judgment": judge(
                    choice["text"], case["expected"], choice["finish_reason"]
                ),
            }
        )
    if (
        sum(r["completion_tokens"] for r in rows)
        != response["usage"]["completion_tokens"]
    ):
        raise ValueError("Batch output token count differs")
    return rows


def api_batch(args: argparse.Namespace) -> None:
    cases = [json.loads(line) for line in args.cases.read_text().splitlines()][
        : args.limit
    ]
    baseline = {r["case_id"]: r for r in json.loads(args.baseline.read_text())}
    args.output_dir.mkdir(parents=True, exist_ok=False)
    save(
        args.output_dir / "protocol.json",
        {
            "model": args.model,
            "case_ids": [c["case_id"] for c in cases],
            "batch_sizes": args.batch_sizes,
            "max_tokens": args.max_tokens,
            "method": (
                "One HTTP completions request carries B token-ID prompts; "
                "nonstream ordering/correctness check"
            ),
        },
    )
    for batch in args.batch_sizes:
        waves, rows = [], []
        begin = time.perf_counter()
        for offset in range(0, len(cases), batch):
            selected = cases[offset : offset + batch]
            payload = {
                "model": args.model,
                "prompt": [c["prompt_token_ids"] for c in selected],
                "max_tokens": args.max_tokens,
                "temperature": 0,
                "seed": 0,
                "logprobs": 1,
                "stream": False,
            }
            request = Request(
                f"{args.base_url}/v1/completions",
                data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"},
            )
            wave = {
                "case_ids": [c["case_id"] for c in selected],
                "start_unix": time.time(),
            }
            started = time.perf_counter()
            try:
                with urlopen(request, timeout=900) as response:
                    result = json.loads(response.read().decode("utf-8"))
                wave["response"] = result
                selected_rows = validate_api_batch(result, selected)
                for row in selected_rows:
                    row["same_as_single_semantic"] = row["judgment"].get(
                        "parsed"
                    ) is not None and row["judgment"].get("parsed") == baseline[
                        row["case_id"]
                    ]["judgment"].get("parsed")
                rows.extend(selected_rows)
                wave["error"] = None
            except (HTTPError, URLError, TimeoutError, OSError, ValueError) as exc:
                detail = (
                    exc.read().decode("utf-8", errors="replace")
                    if isinstance(exc, HTTPError)
                    else str(exc)
                )
                wave["error"] = f"{type(exc).__name__}: {detail}"
            wave["elapsed_s"] = time.perf_counter() - started
            waves.append(wave)
            save(
                args.output_dir / f"batch{batch}.json",
                {"waves": waves, "answers": rows},
            )
            print("API batch", batch, len(rows), "answers", wave["error"], flush=True)
            if wave["error"] is not None:
                raise RuntimeError("Native API batch failed; see saved response")
        save(
            args.output_dir / f"batch{batch}-summary.json",
            {
                "answers": len(rows),
                "correct": sum(r["judgment"]["correct"] for r in rows),
                "same_as_single_semantic": sum(
                    r["same_as_single_semantic"] for r in rows
                ),
                "wall_s": time.perf_counter() - begin,
                "http_batch_e2e_s": percentiles([w["elapsed_s"] for w in waves]),
            },
        )


def load_baseline(path: Path, cases: list[dict], model: str) -> dict:
    protocol = json.loads((path.parent / "protocol.json").read_text())
    if protocol["model"] != model:
        raise ValueError("Single-request baseline belongs to a different model")
    rows = {row["case_id"]: row for row in json.loads(path.read_text())}
    for case in cases:
        row = rows.get(case["case_id"])
        if row is None or row["input_tokens"] != case["input_tokens"]:
            raise ValueError("Single-request baseline does not cover these cases")
        if row["expected"] != case["expected"]:
            raise ValueError("Single-request baseline gold differs")
    return rows


def run(args: argparse.Namespace) -> None:
    cases = [json.loads(line) for line in args.cases.read_text().splitlines()]
    assert len({c["case_id"] for c in cases}) == len(cases)
    assert all(len(c["prompt_token_ids"]) == c["input_tokens"] for c in cases)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    ranks = dict(
        re.findall(
            r"Worker_TP(\d) pid=(\d+)", (args.service_run / "server.log").read_text()
        )
    )
    assert len(ranks) == 4
    protocol = {
        k: str(v) if isinstance(v, Path) else v
        for k, v in vars(args).items()
        if k != "func"
    }
    protocol.update(
        worker_ranks=ranks,
        case_ids=[c["case_id"] for c in cases],
        acceptance=(
            "Controlled Chinese extraction; compare gold and "
            "batch=1 semantic answers. No broad model accuracy claim."
        ),
    )
    save(args.output_dir / "protocol.json", protocol)
    stop = threading.Event()
    stage = {"name": "warmup"}
    metrics = []

    def monitor() -> None:
        with (args.output_dir / "metrics.jsonl").open("w") as file:
            while not stop.is_set():
                sample = {
                    "time_unix": time.time(),
                    "stage": stage["name"],
                    "workers_alive": {
                        r: Path(f"/proc/{p}").exists() for r, p in ranks.items()
                    },
                }
                try:
                    with urlopen(f"{args.base_url}/metrics", timeout=3) as response:
                        raw = response.read().decode()
                    for key in (
                        "num_requests_running",
                        "num_requests_waiting",
                        "kv_cache_usage_perc",
                    ):
                        matches = re.findall(
                            rf"^vllm:{key}\{{[^\n]*\}} ([\d.eE+-]+)", raw, re.M
                        )
                        sample[key] = sum(map(float, matches)) if matches else None
                except (URLError, TimeoutError, OSError) as exc:
                    sample["error"] = str(exc)
                metrics.append(sample)
                file.write(json.dumps(sample) + "\n")
                file.flush()
                stop.wait(0.2)

    thread = threading.Thread(target=monitor, name="acceptance-monitor")
    thread.start()
    baseline = load_baseline(args.baseline, cases, args.model) if args.baseline else {}
    summaries = {}
    try:
        warmups = []
        for index in range(args.warmup):
            row = complete(
                args.base_url, args.model, cases[0], args.max_tokens, args.timeout
            )
            warmups.append(row)
            save(args.output_dir / "warmups.json", warmups)
            print("warmup", index + 1, row["elapsed_s"], row["error"], flush=True)
            if row["error"] is not None:
                break
        if any(row["error"] for row in warmups):
            raise RuntimeError("Warmup failed; see warmups.json")
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=max([*args.batch_sizes, args.stress_concurrency])
        ) as pool:
            for batch in args.batch_sizes:
                stage["name"] = f"batch{batch}"
                rows = []
                begin = time.perf_counter()
                for offset in range(0, len(cases), batch):
                    selected = cases[offset : offset + batch]
                    barrier = threading.Barrier(len(selected))
                    futures = [
                        pool.submit(
                            complete,
                            args.base_url,
                            args.model,
                            c,
                            args.max_tokens,
                            args.timeout,
                            barrier,
                        )
                        for c in selected
                    ]
                    for future in concurrent.futures.as_completed(futures):
                        row = future.result()
                        row["wave"] = offset // batch
                        reference = baseline.get(row["case_id"])
                        row["same_as_single_text"] = (
                            row["text"] == reference["text"] if reference else None
                        )
                        row["same_as_single_semantic"] = (
                            row["judgment"].get("parsed") is not None
                            and row["judgment"].get("parsed")
                            == reference["judgment"].get("parsed")
                            and row["error"] is None
                            and reference["error"] is None
                            if reference
                            else None
                        )
                        rows.append(row)
                        if batch == 1:
                            baseline[row["case_id"]] = row
                        save(args.output_dir / f"batch{batch}-requests.json", rows)
                        print(
                            stage["name"],
                            len(rows),
                            row["case_id"],
                            round(row["elapsed_s"], 3),
                            row["judgment"]["correct"],
                            row["error"],
                            flush=True,
                        )
                    if any(r["error"] for r in rows):
                        summary = summarize(rows, time.perf_counter() - begin)
                        summary["expected_requests"] = len(cases)
                        summary["coverage_complete"] = len(rows) == len(cases)
                        summary["aborted_due_to_error"] = True
                        summaries[stage["name"]] = summary
                        save(args.output_dir / "summary.json", summaries)
                        raise RuntimeError(f"Request failure in batch{batch}")
                summary = summarize(rows, time.perf_counter() - begin)
                summary["expected_requests"] = len(cases)
                summary["coverage_complete"] = len(rows) == len(cases)
                summary["aborted_due_to_error"] = False
                summary["same_as_single_semantic"] = sum(
                    r["same_as_single_semantic"] is True for r in rows
                )
                summaries[stage["name"]] = summary
                save(args.output_dir / "summary.json", summaries)
            if args.stress_seconds > 0:
                stage["name"] = f"stress{args.stress_concurrency}"
                begin = time.perf_counter()
                pending, rows, index = {}, [], 0
                failed_at = None
                while (
                    failed_at is None
                    and time.perf_counter() - begin < args.stress_seconds
                ) or pending:
                    while (
                        failed_at is None
                        and time.perf_counter() - begin < args.stress_seconds
                        and len(pending) < args.stress_concurrency
                    ):
                        case = cases[index % len(cases)]
                        future = pool.submit(
                            complete,
                            args.base_url,
                            args.model,
                            case,
                            args.max_tokens,
                            args.timeout,
                        )
                        pending[future] = index
                        index += 1
                    finished, _ = concurrent.futures.wait(
                        pending, return_when=concurrent.futures.FIRST_COMPLETED
                    )
                    for future in finished:
                        row = future.result()
                        row["sequence"] = pending.pop(future)
                        reference = baseline[row["case_id"]]
                        row["same_as_single_semantic"] = (
                            row["judgment"].get("parsed") is not None
                            and row["judgment"].get("parsed")
                            == reference["judgment"].get("parsed")
                            and row["error"] is None
                            and reference["error"] is None
                        )
                        rows.append(row)
                        save(args.output_dir / "stress-requests.json", rows)
                        print(
                            stage["name"],
                            len(rows),
                            row["case_id"],
                            round(row["elapsed_s"], 3),
                            row["judgment"]["correct"],
                            row["error"],
                            flush=True,
                        )
                    if failed_at is None and any(r["error"] for r in rows):
                        failed_at = time.perf_counter()
                summaries[stage["name"]] = summarize(rows, time.perf_counter() - begin)
                summaries[stage["name"]]["started_requests"] = index
                summaries[stage["name"]]["load_duration_s"] = min(
                    args.stress_seconds,
                    (failed_at or time.perf_counter()) - begin,
                )
                summaries[stage["name"]]["aborted_due_to_error"] = failed_at is not None
                summaries[stage["name"]]["same_as_single_semantic"] = sum(
                    r["same_as_single_semantic"] for r in rows
                )
                save(args.output_dir / "summary.json", summaries)
                if failed_at is not None:
                    raise RuntimeError("Sustained-load request failed")
        with urlopen(f"{args.base_url}/health", timeout=10) as response:
            save(
                args.output_dir / "final_health.json", {"http_status": response.status}
            )
    finally:
        stop.set()
        thread.join(timeout=5)
        for name, summary in summaries.items():
            samples = [m for m in metrics if m["stage"] == name]
            summary["observed_running_max"] = max(
                (
                    m["num_requests_running"]
                    for m in samples
                    if m.get("num_requests_running") is not None
                ),
                default=None,
            )
            summary["all_workers_alive"] = bool(samples) and all(
                all(m["workers_alive"].values()) for m in samples
            )
        save(args.output_dir / "summary.json", summaries)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser("prepare")
    prepare_parser.add_argument("--dataset", type=Path, required=True)
    prepare_parser.add_argument("--tokenizer", type=Path, required=True)
    prepare_parser.add_argument("--output", type=Path, required=True)
    prepare_parser.set_defaults(func=prepare)
    run_parser = commands.add_parser("run")
    run_parser.add_argument("--base-url", default="http://127.0.0.1:8018")
    run_parser.add_argument("--model", required=True)
    run_parser.add_argument("--cases", type=Path, required=True)
    run_parser.add_argument("--service-run", type=Path, required=True)
    run_parser.add_argument("--output-dir", type=Path, required=True)
    run_parser.add_argument("--batch-sizes", nargs="*", type=int, default=[1, 2, 4, 8])
    run_parser.add_argument("--baseline", type=Path)
    run_parser.add_argument("--max-tokens", type=int, default=256)
    run_parser.add_argument("--warmup", type=int, default=3)
    run_parser.add_argument("--timeout", type=float, default=900)
    run_parser.add_argument("--stress-seconds", type=float, default=600)
    run_parser.add_argument("--stress-concurrency", type=int, default=8)
    run_parser.set_defaults(func=run)
    batch_parser = commands.add_parser("api-batch")
    batch_parser.add_argument("--base-url", default="http://127.0.0.1:8018")
    batch_parser.add_argument("--model", required=True)
    batch_parser.add_argument("--cases", type=Path, required=True)
    batch_parser.add_argument("--baseline", type=Path, required=True)
    batch_parser.add_argument("--output-dir", type=Path, required=True)
    batch_parser.add_argument("--batch-sizes", nargs="+", type=int, default=[2, 4, 8])
    batch_parser.add_argument("--max-tokens", type=int, default=256)
    batch_parser.add_argument("--limit", type=int, default=8)
    batch_parser.set_defaults(func=api_batch)
    args = parser.parse_args()
    if args.command == "run":
        if args.batch_sizes and min(args.batch_sizes) < 1:
            parser.error("batch-sizes must contain positive values")
        if args.baseline is None and (not args.batch_sizes or args.batch_sizes[0] != 1):
            parser.error("start with batch 1 or provide --baseline")
        if args.warmup < 0 or args.stress_seconds < 0 or args.stress_concurrency < 1:
            parser.error("invalid warmup/stress settings")
        if not args.batch_sizes and not args.stress_seconds:
            parser.error("select batch groups or a positive stress duration")
    args.func(args)


if __name__ == "__main__":
    main()
