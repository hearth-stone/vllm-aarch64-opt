#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Copy the harness MMLU tasks and point them at a local dataset checkout."""

import argparse
import json
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--harness-dir", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--task-dir", type=Path, required=True)
    args = parser.parse_args()

    source = args.harness_dir / "lm_eval/tasks/mmlu/default"
    dataset = args.dataset_dir.resolve()
    template_source = source / "_default_template_yaml"
    sample = dataset / "abstract_algebra/test-00000-of-00001.parquet"
    if not template_source.is_file():
        parser.error(f"MMLU task template missing: {template_source}")
    if not sample.is_file() or sample.stat().st_size == 0:
        parser.error(f"MMLU dataset file missing or empty: {sample}")

    old = "dataset_path: cais/mmlu"
    content = template_source.read_text(encoding="utf-8")
    if content.count(old) != 1:
        parser.error(f"expected one '{old}' in {template_source}")

    shutil.copytree(source, args.task_dir, dirs_exist_ok=True)
    template = args.task_dir / "_default_template_yaml"
    template.write_text(
        content.replace(old, f"dataset_path: {json.dumps(str(dataset))}"),
        encoding="utf-8",
    )
    print(f"MMLU tasks: {args.task_dir}")
    print(f"Dataset: {dataset}")


if __name__ == "__main__":
    main()
