# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare choice-constrained MMLU single answers with one native API batch."""

import argparse
import json
import math
import statistics
import time
from pathlib import Path
from urllib.request import Request, urlopen


def save(path: Path, value: object) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--cases-dir", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8018")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.cases_dir
    cases = list(map(json.loads, (root / "cases32.jsonl").read_text().splitlines()))
    metadata = json.loads((root / "dataset-metadata.json").read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    save(
        args.output / "protocol.json",
        {
            "base_url": args.base_url,
            "model": args.model,
            "max_tokens": 1,
            "temperature": 0,
            "seed": 0,
            "allowed_token_ids": metadata["allowed_token_ids"],
            "native_batch_size": 32,
            "single_requests": 32,
            "warmups": 2,
            "metric": "prefill plus one answer token, no subsequent decode",
        },
    )

    def complete(selected: list[dict], phase: str) -> dict:
        payload = {
            "model": args.model,
            "prompt": [c["prompt_token_ids"] for c in selected],
            "temperature": 0,
            "seed": 0,
            "max_tokens": 1,
            "logprobs": 4,
            "allowed_token_ids": metadata["allowed_token_ids"],
            "stream": False,
        }
        start = time.time()
        record = {
            "phase": phase,
            "start_unix": start,
            "case_ids": [c["case_id"] for c in selected],
            "error": None,
        }
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
            assert usage["prompt_tokens"] == sum(c["input_tokens"] for c in selected), (
                usage
            )
            assert usage["completion_tokens"] == len(selected), usage
            assert usage["total_tokens"] == usage["prompt_tokens"] + len(selected), (
                usage
            )
            choices = sorted(result["choices"], key=lambda c: c["index"])
            assert [c["index"] for c in choices] == list(range(len(selected)))
            rows = []
            for case, choice in zip(selected, choices):
                logs = choice["logprobs"]["token_logprobs"]
                assert len(logs) == 1 and math.isfinite(logs[0]), choice
                assert len(choice["logprobs"]["tokens"]) == 1, choice
                answer = choice["text"]
                assert answer in "ABCD" and len(answer) == 1, choice
                assert choice["finish_reason"] in ("length", "stop"), choice
                rows.append(
                    {
                        "case_id": case["case_id"],
                        "answer": answer,
                        "gold": case["gold"],
                        "correct": answer == case["gold"],
                        "input_tokens": case["input_tokens"],
                        "logprob": logs[0],
                    }
                )
            record["rows"] = rows
        except (OSError, ValueError, KeyError, TypeError, AssertionError) as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
        record.update(end_unix=time.time(), elapsed_s=time.time() - start)
        return record

    for i in range(2):
        row = complete(cases[:1], "warmup")
        save(args.output / f"warmup{i}.json", row)
        assert row["error"] is None, row["error"]
    singles = []
    for case in cases:
        record = complete([case], "single")
        singles.append(record)
        save(args.output / "single.json", singles)
        print(
            "single",
            len(singles),
            record["elapsed_s"],
            record.get("rows"),
            record["error"],
            flush=True,
        )
        assert record["error"] is None, record["error"]
    batch = complete(cases, "batch32")
    save(args.output / "batch32.json", batch)
    assert batch["error"] is None, batch["error"]
    baseline = {r["rows"][0]["case_id"]: r["rows"][0] for r in singles}
    differences = [
        r["case_id"]
        for r in batch["rows"]
        if r["answer"] != baseline[r["case_id"]]["answer"]
    ]
    summary = {
        "questions": len(cases),
        "single_correct": sum(r["rows"][0]["correct"] for r in singles),
        "batch_correct": sum(r["correct"] for r in batch["rows"]),
        "same_as_single": len(cases) - len(differences),
        "different_case_ids": differences,
        "single_mean_s": statistics.mean(r["elapsed_s"] for r in singles),
        "single_total_s": sum(r["elapsed_s"] for r in singles),
        "batch_total_s": batch["elapsed_s"],
        "batch_tokens_per_s": metadata["sum_tokens"] / batch["elapsed_s"],
        "all_responses_valid": True,
        "batch_consistent": not differences,
    }
    repeat_singles = []
    suspect = [
        c
        for c in cases
        if c["case_id"] in set(differences) | {"mmlu-geography-test-014"}
    ]
    for round_index in range(2):
        for case in suspect:
            record = complete([case], "single-repeat")
            repeat_singles.append(record)
            save(args.output / "repeat-singles.json", repeat_singles)
            assert record["error"] is None, record["error"]
    repeat_batch = complete(cases, "batch32-repeat")
    save(args.output / "repeat-batch32.json", repeat_batch)
    assert repeat_batch["error"] is None, repeat_batch["error"]
    summary["batch_repeat_same_as_first"] = sum(
        a["answer"] == b["answer"] for a, b in zip(batch["rows"], repeat_batch["rows"])
    )
    summary["repeated_single_answers"] = {
        c["case_id"]: [
            r["rows"][0]["answer"]
            for r in repeat_singles
            if r["rows"][0]["case_id"] == c["case_id"]
        ]
        for c in suspect
    }
    save(args.output / "summary.json", summary)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
