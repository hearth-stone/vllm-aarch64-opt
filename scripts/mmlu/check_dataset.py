#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Verify every MMLU Parquet file was downloaded and load one subject offline."""

import argparse
import os
import subprocess
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    args = parser.parse_args()
    dataset = args.dataset_dir.resolve()
    result = subprocess.run(
        ["git", "-C", str(dataset), "ls-files", "--", "*.parquet"],
        check=True,
        capture_output=True,
        text=True,
    )
    files = [dataset / name for name in result.stdout.splitlines()]
    if not files:
        parser.error(f"no tracked MMLU Parquet files in {dataset}")
    missing = []
    for path in files:
        if not path.is_file() or path.stat().st_size < 8:
            missing.append(path)
            continue
        with path.open("rb") as stream:
            if stream.read(4) != b"PAR1":
                missing.append(path)
    if missing:
        parser.error(
            f"{len(missing)} Parquet files are missing or still Git LFS pointers; "
            f"first: {missing[0]}"
        )

    os.environ["HF_DATASETS_OFFLINE"] = "1"
    from datasets import load_dataset

    data = load_dataset(str(dataset), "abstract_algebra")
    if not data["dev"] or not data["test"]:
        parser.error("abstract_algebra dev or test split is empty")
    print(f"Validated {len(files)} Parquet files in {dataset}")
    print(
        f"abstract_algebra: {dict((split, len(rows)) for split, rows in data.items())}"
    )


if __name__ == "__main__":
    main()
