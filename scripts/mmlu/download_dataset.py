#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Download the pinned MMLU checkout and replace skipped Git LFS pointers."""

import argparse
import os
import subprocess
from pathlib import Path


def tracked_parquet(dataset: Path) -> list[str]:
    result = subprocess.run(
        ["git", "-C", str(dataset), "ls-files", "--", "*.parquet"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.splitlines()


def missing_parquet(dataset: Path, paths: list[str]) -> list[str]:
    missing = []
    for relative in paths:
        path = dataset / relative
        if not path.is_file() or path.stat().st_size < 8:
            missing.append(relative)
            continue
        with path.open("rb") as stream:
            if stream.read(4) != b"PAR1":
                missing.append(relative)
    return missing


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--hf-cli", type=Path, required=True)
    parser.add_argument("--endpoint", default="https://hf-mirror.com")
    args = parser.parse_args()
    dataset = args.dataset_dir.resolve()
    if not args.hf_cli.is_file():
        parser.error(f"hf executable missing: {args.hf_cli}")
    paths = tracked_parquet(dataset)
    if not paths:
        parser.error(f"no tracked MMLU Parquet files in {dataset}")
    revision = subprocess.run(
        ["git", "-C", str(dataset), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    command = [
        str(args.hf_cli),
        "download",
        "cais/mmlu",
        "--repo-type",
        "dataset",
        "--revision",
        revision,
        "--local-dir",
        str(dataset),
    ]
    env = {**os.environ, "HF_ENDPOINT": args.endpoint, "HF_HUB_DISABLE_XET": "1"}
    print(f"Downloading MMLU revision {revision} from {args.endpoint}", flush=True)
    subprocess.run(command, check=False, env=env)

    for relative in missing_parquet(dataset, paths):
        print(f"Replacing Git LFS pointer: {relative}", flush=True)
        for _ in range(3):
            subprocess.run(
                command[:3] + [relative] + command[3:] + ["--force-download"],
                check=False,
                env=env,
            )
            if relative not in missing_parquet(dataset, [relative]):
                break
    missing = missing_parquet(dataset, paths)
    if missing:
        parser.error(
            f"{len(missing)} MMLU Parquet files remain unavailable: {missing[:5]}"
        )
    print(f"Downloaded and verified {len(paths)} MMLU Parquet files")


if __name__ == "__main__":
    main()
