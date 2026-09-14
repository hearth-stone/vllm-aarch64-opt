#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Run exact-token prefill cases sequentially in one vLLM engine."""

from __future__ import annotations

import argparse
import csv
import json
import os
import threading
import time
from pathlib import Path

import numpy as np
import regex as re

from vllm import LLM, SamplingParams, TokensPrompt
from vllm.config import ProfilerConfig
from vllm.model_executor.layers.fused_moe.file_route_recorder import (
    ROUTE_CAPTURE_CONTROL_ENV,
    ROUTE_CAPTURE_DIR_ENV,
    write_route_control,
)
from vllm.outputs import RequestOutput


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--cases", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tensor-parallel-size", type=int, default=4)
    parser.add_argument("--max-model-len", type=int, default=2100)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.4)
    parser.add_argument("--kv-cache-memory-bytes", type=int)
    parser.add_argument("--numa-bind-cpus")
    parser.add_argument("--numa-bind-nodes")
    parser.add_argument("--expected-prompt-tokens", type=int, default=2048)
    parser.add_argument("--min-measured-cases", type=int, default=10)
    parser.add_argument("--request-interval-seconds", type=float, default=0.0)
    parser.add_argument(
        "--profile-dir",
        help="Profile all measured requests in one worker trace per TP rank.",
    )
    parser.add_argument("--warmup-case-index", type=int, default=0)
    parser.add_argument(
        "--repeat-measured-case-index",
        type=int,
        help="Measure only this case, repeated --repeat-measured-case-count times.",
    )
    parser.add_argument("--repeat-measured-case-count", type=int, default=10)
    parser.add_argument(
        "--exclude-warmup-case",
        action="store_true",
        help="Do not run the warmup case again as a measured request.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-tokens", type=int, default=1)
    parser.add_argument(
        "--frequency-probe-cpus",
        help="Pipe-separated CPU-id groups sampled from scaling_cur_freq.",
    )
    parser.add_argument("--frequency-probe-interval-seconds", type=float, default=0.05)
    parser.add_argument(
        "--thermal-probe-zones",
        help="Comma-separated thermal zone ids sampled after each request.",
    )
    parser.add_argument(
        "--route-capture-dir",
        help=(
            "Capture prompt-token MoE expert IDs as one compressed NPZ per case "
            "and write aggregate route summaries."
        ),
    )
    return parser.parse_args()


def split(value: str | None, separator: str) -> list[str] | None:
    if value is None:
        return None
    return [item.strip() for item in value.split(separator) if item.strip()]


def load_cases(path: str, expected_tokens: int) -> list[dict[str, object]]:
    cases = []
    with open(path, encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            if not line.strip():
                continue
            case = json.loads(line)
            token_ids = case.get("prompt_token_ids")
            if not isinstance(token_ids, list) or len(token_ids) != expected_tokens:
                raise ValueError(
                    f"line {line_number}: expected {expected_tokens} prompt tokens"
                )
            cases.append(case)
    if not cases:
        raise ValueError("at least one case is required")
    return cases


def build_profiler_config(profile_dir: str) -> ProfilerConfig:
    profile_path = os.path.abspath(profile_dir)
    os.makedirs(profile_path, exist_ok=True)
    return ProfilerConfig(
        profiler="torch",
        torch_profiler_dir=profile_path,
        torch_profiler_with_stack=False,
        torch_profiler_with_memory=False,
        torch_profiler_record_shapes=False,
        torch_profiler_with_flops=False,
        torch_profiler_use_gzip=False,
        torch_profiler_dump_cuda_time_total=False,
        ignore_frontend=True,
        delay_iterations=0,
        max_iterations=0,
        warmup_iterations=0,
        active_iterations=1,
        wait_iterations=0,
    )


def route_output_name(
    phase: str,
    request_index: int,
    source_case_index: int,
    case_id: object,
) -> str:
    safe_case_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(case_id))
    return (
        f"{phase}_request{request_index:03d}_case{source_case_index:03d}_"
        f"{safe_case_id}.npz"
    )


def run_with_route_capture(
    llm: LLM,
    prompt: TokensPrompt,
    sampling_params: SamplingParams,
    *,
    route_dir: Path | None,
    control_path: Path | None,
    phase: str,
    request_index: int,
    source_case_index: int,
    case: dict[str, object],
) -> tuple[list[RequestOutput], Path | None]:
    route_path = None
    if route_dir is not None:
        assert control_path is not None
        output_name = route_output_name(
            phase,
            request_index,
            source_case_index,
            case["case_id"],
        )
        route_path = route_dir / output_name
        route_path.unlink(missing_ok=True)
        write_route_control(
            control_path,
            {
                "capture_id": f"{phase}:{request_index}:{source_case_index}",
                "output_file": output_name,
                "phase": phase,
                "request_index": request_index,
                "source_case_index": source_case_index,
                "case_id": case["case_id"],
                "title": case["title"],
                "prompt_token_ids": case["prompt_token_ids"],
            },
        )
    try:
        outputs = llm.generate([prompt], sampling_params)
    finally:
        if control_path is not None:
            control_path.unlink(missing_ok=True)
    if route_path is not None and not route_path.is_file():
        raise RuntimeError(f"expected MoE route capture file {route_path}")
    return outputs, route_path


def summarize_route_captures(
    route_dir: Path,
    manifest_rows: list[dict[str, object]],
    expected_tokens: int,
) -> dict[str, object]:
    route_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = route_dir / "route_manifest.jsonl"
    counts_by_case = None
    aggregate_counts = None
    route_shape = None
    with manifest_path.open("w", encoding="utf-8") as manifest_file:
        for row_index, row in enumerate(manifest_rows):
            route_path = route_dir / str(row["route_file"])
            with np.load(route_path, allow_pickle=False) as payload:
                expert_ids = payload["expert_ids"]
                prompt_token_ids = payload["prompt_token_ids"]
                shared_expert_ids = payload["shared_expert_ids"]
            if expert_ids.shape[0] != expected_tokens or expert_ids.ndim != 3:
                raise ValueError(
                    f"unexpected expert route shape {expert_ids.shape} in {route_path}"
                )
            if prompt_token_ids.shape != (expected_tokens,):
                raise ValueError(
                    f"unexpected prompt token shape {prompt_token_ids.shape}"
                )
            if route_shape is None:
                route_shape = expert_ids.shape
                num_cases = len(manifest_rows)
                counts_by_case = np.zeros(
                    (num_cases, route_shape[1], 256), dtype=np.int64
                )
                aggregate_counts = np.zeros((route_shape[1], 256), dtype=np.int64)
            elif expert_ids.shape != route_shape:
                raise ValueError(
                    f"route shape changed from {route_shape} to {expert_ids.shape}"
                )
            assert counts_by_case is not None
            assert aggregate_counts is not None
            if expert_ids.size and (expert_ids.min() < 0 or expert_ids.max() >= 256):
                raise ValueError(f"invalid expert ID in {route_path}")
            for layer in range(expert_ids.shape[1]):
                counts = np.bincount(expert_ids[:, layer, :].reshape(-1), minlength=256)
                counts_by_case[row_index, layer] = counts
                aggregate_counts[layer] += counts
            row["shape"] = list(expert_ids.shape)
            row["shared_expert_ids"] = shared_expert_ids.tolist()
            manifest_file.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            )

    assert route_shape is not None
    assert counts_by_case is not None
    assert aggregate_counts is not None
    np.save(route_dir / "expert_counts_by_case.npy", counts_by_case)
    np.save(route_dir / "expert_counts_by_layer.npy", aggregate_counts)
    with (route_dir / "expert_counts_by_layer.csv").open(
        "w", encoding="utf-8", newline=""
    ) as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(("layer", "expert_id", "token_assignments"))
        for layer, layer_counts in enumerate(aggregate_counts):
            for expert_id, count in enumerate(layer_counts):
                writer.writerow((layer, expert_id, int(count)))

    assignments_per_layer = route_shape[0] * route_shape[2] * len(manifest_rows)
    summary = {
        "schema_version": 1,
        "case_count": len(manifest_rows),
        "route_shape_per_case": list(route_shape),
        "routed_expert_count": 256,
        "shared_expert_ids": manifest_rows[0]["shared_expert_ids"],
        "assignments_per_layer": assignments_per_layer,
        "all_layers_complete": bool(
            np.all(aggregate_counts.sum(axis=1) == assignments_per_layer)
        ),
        "top_experts_by_layer": [
            [
                {"expert_id": int(expert_id), "count": int(layer_counts[expert_id])}
                for expert_id in np.argsort(layer_counts)[-10:][::-1]
            ]
            for layer_counts in aggregate_counts
        ],
        "manifest": str(manifest_path),
    }
    with (route_dir / "route_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2, sort_keys=True)
        file.write("\n")
    return {
        "case_count": summary["case_count"],
        "route_shape_per_case": summary["route_shape_per_case"],
        "assignments_per_layer": summary["assignments_per_layer"],
        "all_layers_complete": summary["all_layers_complete"],
        "summary_file": str(route_dir / "route_summary.json"),
    }


def probe_cpu_frequencies(
    cpu_groups: list[list[int]],
    interval_s: float,
    stop: threading.Event,
    samples: list[list[float]],
) -> None:
    paths = [
        [
            Path(f"/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_cur_freq")
            for cpu in group
        ]
        for group in cpu_groups
    ]
    while not stop.is_set():
        group_means = []
        for group_paths in paths:
            values = [float(path.read_text().strip()) / 1000 for path in group_paths]
            group_means.append(float(np.mean(values)))
        samples.append(group_means)
        stop.wait(interval_s)


def main() -> None:
    args = parse_args()
    cases = load_cases(args.cases, args.expected_prompt_tokens)
    if not 0 <= args.warmup_case_index < len(cases):
        raise ValueError("--warmup-case-index is out of range")
    if args.max_tokens <= 0:
        raise ValueError("--max-tokens must be positive")

    rank_cpus = split(args.numa_bind_cpus, "|")
    rank_nodes_text = split(args.numa_bind_nodes, ",")
    rank_nodes = [int(node) for node in rank_nodes_text] if rank_nodes_text else None
    llm_kwargs: dict[str, object] = {
        "model": args.model,
        "seed": args.seed,
        "max_model_len": args.max_model_len,
        "tensor_parallel_size": args.tensor_parallel_size,
        "enforce_eager": True,
        "trust_remote_code": True,
        "generation_config": "vllm",
        "enable_chunked_prefill": False,
        "enable_prefix_caching": False,
        "block_size": args.block_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
    }
    if rank_cpus is not None or rank_nodes is not None:
        if rank_cpus is None or rank_nodes is None:
            raise ValueError("both NUMA CPU and node lists are required")
        if len(rank_cpus) != args.tensor_parallel_size:
            raise ValueError("one NUMA CPU list is required per TP rank")
        if len(rank_nodes) != args.tensor_parallel_size:
            raise ValueError("one NUMA node is required per TP rank")
        llm_kwargs.update(
            numa_bind=True,
            numa_bind_cpus=rank_cpus,
            numa_bind_nodes=rank_nodes,
        )
    if args.kv_cache_memory_bytes is not None:
        llm_kwargs["kv_cache_memory_bytes"] = args.kv_cache_memory_bytes
    if args.profile_dir:
        llm_kwargs["profiler_config"] = build_profiler_config(args.profile_dir)

    route_dir = (
        Path(args.route_capture_dir).resolve() if args.route_capture_dir else None
    )
    control_path = route_dir / "capture_control.json" if route_dir else None
    if route_dir is not None:
        route_dir.mkdir(parents=True, exist_ok=True)
        assert control_path is not None
        control_path.unlink(missing_ok=True)
        os.environ[ROUTE_CAPTURE_DIR_ENV] = str(route_dir)
        os.environ[ROUTE_CAPTURE_CONTROL_ENV] = str(control_path)

    print(f"model={args.model}")
    print(f"cases={len(cases)} x {args.expected_prompt_tokens} tokens")
    profile_status = (
        "enabled:" + os.path.abspath(args.profile_dir)
        if args.profile_dir
        else "disabled"
    )
    print(f"profile={profile_status}")
    print(f"route_capture={route_dir if route_dir is not None else 'disabled'}")
    llm = LLM(**llm_kwargs)
    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=args.max_tokens,
        ignore_eos=True,
        repetition_penalty=1.1,
        seed=args.seed,
    )

    warmup = cases[args.warmup_case_index]
    warmup_start = time.perf_counter_ns()
    _, warmup_route_path = run_with_route_capture(
        llm,
        TokensPrompt(prompt_token_ids=warmup["prompt_token_ids"]),
        sampling_params,
        route_dir=route_dir,
        control_path=control_path,
        phase="warmup",
        request_index=0,
        source_case_index=args.warmup_case_index,
        case=warmup,
    )
    warmup_s = (time.perf_counter_ns() - warmup_start) / 1e9
    print(f"warmup case={warmup['case_id']} duration_s={warmup_s:.6f} excluded=true")
    if args.request_interval_seconds < 0:
        raise ValueError("--request-interval-seconds must be non-negative")

    if args.repeat_measured_case_index is not None:
        if not 0 <= args.repeat_measured_case_index < len(cases):
            raise ValueError("--repeat-measured-case-index is out of range")
        if args.repeat_measured_case_count <= 0:
            raise ValueError("--repeat-measured-case-count must be positive")
        measurement_cases = [
            (args.repeat_measured_case_index, cases[args.repeat_measured_case_index])
            for _ in range(args.repeat_measured_case_count)
        ]
    else:
        measurement_cases = [
            (index, case)
            for index, case in enumerate(cases)
            if not args.exclude_warmup_case or index != args.warmup_case_index
        ]
    if len(measurement_cases) < args.min_measured_cases:
        raise ValueError(
            f"at least {args.min_measured_cases} measured cases are required, "
            f"got {len(measurement_cases)}"
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    durations = []
    route_manifest_rows: list[dict[str, object]] = []
    if warmup_route_path is not None:
        route_manifest_rows.append(
            {
                "phase": "warmup",
                "request_index": 0,
                "source_case_index": args.warmup_case_index,
                "case_id": warmup["case_id"],
                "title": warmup["title"],
                "route_file": warmup_route_path.name,
            }
        )
    frequency_cpu_groups = None
    if args.frequency_probe_cpus:
        frequency_cpu_groups = [
            [int(cpu) for cpu in group.split(",")]
            for group in args.frequency_probe_cpus.split("|")
        ]
        if args.frequency_probe_interval_seconds <= 0:
            raise ValueError("--frequency-probe-interval-seconds must be positive")
    thermal_zones = (
        [int(zone) for zone in args.thermal_probe_zones.split(",")]
        if args.thermal_probe_zones
        else None
    )
    if args.profile_dir:
        llm.start_profile()
    try:
        with output.open("w", encoding="utf-8") as f:
            for index, (source_case_index, case) in enumerate(measurement_cases):
                if args.request_interval_seconds:
                    time.sleep(args.request_interval_seconds)
                frequency_samples: list[list[float]] = []
                frequency_stop = threading.Event()
                frequency_thread = None
                if frequency_cpu_groups is not None:
                    frequency_thread = threading.Thread(
                        target=probe_cpu_frequencies,
                        args=(
                            frequency_cpu_groups,
                            args.frequency_probe_interval_seconds,
                            frequency_stop,
                            frequency_samples,
                        ),
                        daemon=True,
                    )
                    frequency_thread.start()
                start_ns = time.perf_counter_ns()
                outputs, route_path = run_with_route_capture(
                    llm,
                    TokensPrompt(prompt_token_ids=case["prompt_token_ids"]),
                    sampling_params,
                    route_dir=route_dir,
                    control_path=control_path,
                    phase="measured",
                    request_index=index,
                    source_case_index=source_case_index,
                    case=case,
                )
                end_ns = time.perf_counter_ns()
                if frequency_thread is not None:
                    frequency_stop.set()
                    frequency_thread.join()
                duration_s = (end_ns - start_ns) / 1e9
                durations.append(duration_s)
                completion = outputs[0].outputs[0]
                generated_token_ids = list(completion.token_ids)
                generated_tokens = len(generated_token_ids)
                row = {
                    "request_index": index,
                    "case_id": case["case_id"],
                    "title": case["title"],
                    "prompt_tokens": len(case["prompt_token_ids"]),
                    "generated_tokens": generated_tokens,
                    "generated_token_ids": generated_token_ids,
                    "generated_text": completion.text,
                    "duration_s": duration_s,
                    "input_tokens_per_s": len(case["prompt_token_ids"]) / duration_s,
                }
                if route_path is not None:
                    row["route_capture_file"] = route_path.name
                    route_manifest_rows.append(
                        {
                            "phase": "measured",
                            "request_index": index,
                            "source_case_index": source_case_index,
                            "case_id": case["case_id"],
                            "title": case["title"],
                            "route_file": route_path.name,
                        }
                    )
                if frequency_samples:
                    frequency_array = np.asarray(frequency_samples)
                    row["frequency_mhz"] = [
                        {
                            "mean": float(np.mean(frequency_array[:, group_index])),
                            "min": float(np.min(frequency_array[:, group_index])),
                            "max": float(np.max(frequency_array[:, group_index])),
                        }
                        for group_index in range(frequency_array.shape[1])
                    ]
                if thermal_zones is not None:
                    row["thermal_c"] = [
                        float(
                            Path(
                                f"/sys/class/thermal/thermal_zone{zone}/temp"
                            ).read_text()
                        )
                        / 1000
                        for zone in thermal_zones
                    ]
                f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                f.flush()
                print(
                    f"request={index} case={case['case_id']} title={case['title']} "
                    f"duration_s={duration_s:.6f}"
                )
                if frequency_samples:
                    means = ",".join(
                        f"group{group_index}={values['mean']:.1f}MHz"
                        for group_index, values in enumerate(row["frequency_mhz"])
                    )
                    print(f"frequency request={index} {means}")
                if thermal_zones is not None:
                    temperatures = ",".join(
                        f"zone{zone}={temperature:.1f}C"
                        for zone, temperature in zip(
                            thermal_zones, row["thermal_c"], strict=True
                        )
                    )
                    print(f"thermal request={index} {temperatures}")
    finally:
        if args.profile_dir:
            llm.stop_profile()

    route_summary = None
    if route_dir is not None:
        route_summary = summarize_route_captures(
            route_dir,
            route_manifest_rows,
            args.expected_prompt_tokens,
        )

    summary = {
        "schema_version": 1,
        "model": args.model,
        "cases_file": os.path.abspath(args.cases),
        "sample_count": len(durations),
        "prompt_tokens_per_case": args.expected_prompt_tokens,
        "max_tokens": args.max_tokens,
        "warmup_case_id": warmup["case_id"],
        "warmup_duration_s": warmup_s,
        "warmup_excluded": True,
        "request_interval_seconds": args.request_interval_seconds,
        "profile_dir": os.path.abspath(args.profile_dir) if args.profile_dir else None,
        "route_capture_dir": str(route_dir) if route_dir is not None else None,
        "route_capture_summary": route_summary,
        "avg_duration_s": float(np.mean(durations)),
        "p95_duration_s": float(np.percentile(durations, 95, method="linear")),
        "min_duration_s": float(np.min(durations)),
        "max_duration_s": float(np.max(durations)),
        "avg_input_tokens_per_s": float(
            args.expected_prompt_tokens / np.mean(durations)
        ),
        "p95_method": "numpy linear interpolation",
    }
    summary_path = output.with_suffix(output.suffix + ".summary.json")
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
