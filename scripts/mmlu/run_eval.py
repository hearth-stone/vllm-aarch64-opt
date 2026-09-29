#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run the MMLU smoke or full evaluation and save the run provenance."""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def git_commit(path: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lm-eval", type=Path, required=True)
    parser.add_argument("--harness-dir", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--task-dir", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8004")
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("smoke", "full"), required=True)
    args = parser.parse_args()

    if not args.lm_eval.is_file():
        parser.error(f"lm-eval executable missing: {args.lm_eval}")
    if not (args.task_dir / "_default_template_yaml").is_file():
        parser.error(f"prepared MMLU tasks missing: {args.task_dir}")

    model_args = (
        f"model={args.model},base_url={args.base_url.rstrip('/')}/v1/completions,"
        "tokenizer_backend=remote,num_concurrent=1,timeout=3600,"
        f"max_retries=3,max_length={args.max_length}"
    )
    command = [
        str(args.lm_eval),
        "--model",
        "local-completions",
        "--model_args",
        model_args,
        "--tasks",
        "mmlu_abstract_algebra" if args.mode == "smoke" else "mmlu",
        "--include_path",
        str(args.task_dir),
        "--num_fewshot",
        "5",
        "--batch_size",
        "1",
    ]
    if args.mode == "smoke":
        command += ["--limit", "2"]
    command += [
        "--output_path",
        str(args.output_dir / args.mode),
        "--log_samples",
    ]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "harness.commit").write_text(
        git_commit(args.harness_dir) + "\n", encoding="utf-8"
    )
    (args.output_dir / "dataset.commit").write_text(
        git_commit(args.dataset_dir) + "\n", encoding="utf-8"
    )
    log_path = args.output_dir / f"{args.mode}.log"
    env = {**os.environ, "HF_DATASETS_OFFLINE": "1"}
    print(f"Running {args.mode} MMLU; log: {log_path}", flush=True)
    with (
        subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
            bufsize=1,
            cwd=args.harness_dir,
        ) as process,
        log_path.open("w", encoding="utf-8") as log,
    ):
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log.write(line)
            log.flush()
        return_code = process.wait()
    if return_code:
        raise SystemExit(return_code)


if __name__ == "__main__":
    main()
