# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sustain native batch32 requests and retain validity and answer drift evidence."""

import argparse
import json
import math
import statistics
import time
from pathlib import Path
from urllib.request import Request, urlopen

from compare import save


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--cases-dir", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8018")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds", type=int, default=600)
    args = parser.parse_args()
    if args.seconds <= 0:
        parser.error("--seconds must be positive")
    root = args.cases_dir
    cases = list(map(json.loads, (root / "cases32.jsonl").read_text().splitlines()))
    metadata = json.loads((root / "dataset-metadata.json").read_text())
    baseline = args.baseline
    baseline_protocol = json.loads((baseline / "protocol.json").read_text())
    if baseline_protocol["model"] != args.model:
        parser.error("--baseline must come from the same served model")
    single = {
        r["rows"][0]["case_id"]: r["rows"][0]["answer"]
        for r in json.loads((baseline / "single.json").read_text())
    }
    batch = {
        r["case_id"]: r["answer"]
        for r in json.loads((baseline / "batch32.json").read_text())["rows"]
    }
    payload = {
        "model": args.model,
        "prompt": [c["prompt_token_ids"] for c in cases],
        "temperature": 0,
        "seed": 0,
        "max_tokens": 1,
        "logprobs": 4,
        "allowed_token_ids": metadata["allowed_token_ids"],
        "stream": False,
    }
    args.output.mkdir(parents=True, exist_ok=False)
    save(
        args.output / "protocol.json",
        {
            "base_url": args.base_url,
            "model": args.model,
            "seconds": args.seconds,
            "batch_size": 32,
            "gap_s": 0,
            "max_model_len": 8192,
            "max_num_batched_tokens": 8192,
            "max_tokens": 1,
            "payload": payload,
            "baseline_directory": str(baseline),
            "warmup": "first wave excluded from timed stress",
        },
    )
    rows = []
    started = None
    deadline = None
    while deadline is None or time.monotonic() < deadline:
        wave = len(rows)
        record = {
            "wave": wave,
            "warmup": wave == 0,
            "start_unix": time.time(),
            "error": None,
        }
        begin = time.monotonic()
        try:
            request = Request(
                f"{args.base_url.rstrip('/')}/v1/completions",
                data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"},
            )
            with urlopen(request, timeout=600) as response:
                result = json.load(response)
            record["response"] = result
            usage = result["usage"]
            assert usage["prompt_tokens"] == metadata["sum_tokens"], usage
            assert (
                usage["completion_tokens"] == 32
                and usage["total_tokens"] == metadata["sum_tokens"] + 32
            ), usage
            choices = sorted(result["choices"], key=lambda c: c["index"])
            assert [c["index"] for c in choices] == list(range(32))
            answers = []
            for case, choice in zip(cases, choices):
                logprobs = choice["logprobs"]
                assert len(logprobs["tokens"]) == len(logprobs["token_logprobs"]) == 1
                assert math.isfinite(logprobs["token_logprobs"][0])
                assert choice["text"] in "ABCD" and len(choice["text"]) == 1
                assert choice["finish_reason"] in ("length", "stop")
                answer = choice["text"]
                answers.append(
                    {
                        "case_id": case["case_id"],
                        "answer": answer,
                        "gold": case["gold"],
                        "correct": answer == case["gold"],
                        "same_as_single": answer == single[case["case_id"]],
                        "same_as_previous_batch": answer == batch[case["case_id"]],
                    }
                )
            record["answers"] = answers
        except (OSError, ValueError, KeyError, TypeError, AssertionError) as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
        record.update(end_unix=time.time(), elapsed_s=time.monotonic() - begin)
        rows.append(record)
        save(args.output / f"wave-{wave:03d}.json", record)
        print(
            wave,
            round(record["elapsed_s"], 3),
            record["error"],
            None
            if record["error"]
            else sum(r["same_as_single"] for r in record["answers"]),
            flush=True,
        )
        if record["error"]:
            break
        if started is None:
            started = time.monotonic()
            deadline = started + args.seconds
    timed = [r for r in rows if not r["warmup"]]
    valid = [r for r in timed if r["error"] is None]
    elapsed = time.monotonic() - started if started is not None else 0
    summary = {
        "model": args.model,
        "target_s": args.seconds,
        "measured_s": elapsed,
        "duration_reached": elapsed >= args.seconds,
        "timed_waves": len(timed),
        "completed_questions": 32 * len(valid),
        "failed_waves": sum(r["error"] is not None for r in rows),
        "mean_batch_s": statistics.mean(r["elapsed_s"] for r in valid)
        if valid
        else None,
        "min_batch_s": min((r["elapsed_s"] for r in valid), default=None),
        "max_batch_s": max((r["elapsed_s"] for r in valid), default=None),
        "prompt_tokens_per_s": metadata["sum_tokens"] * len(valid) / elapsed
        if elapsed
        else 0,
        "gold_correct": sum(a["correct"] for r in valid for a in r["answers"]),
        "same_as_single": sum(a["same_as_single"] for r in valid for a in r["answers"]),
        "same_as_previous_batch": sum(
            a["same_as_previous_batch"] for r in valid for a in r["answers"]
        ),
        "different_from_single_case_ids": sorted(
            {
                a["case_id"]
                for r in valid
                for a in r["answers"]
                if not a["same_as_single"]
            }
        ),
        "different_from_previous_batch_case_ids": sorted(
            {
                a["case_id"]
                for r in valid
                for a in r["answers"]
                if not a["same_as_previous_batch"]
            }
        ),
    }
    save(args.output / "summary.json", summary)
    print(json.dumps(summary), flush=True)
    if summary["failed_waves"] or not summary["duration_reached"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
