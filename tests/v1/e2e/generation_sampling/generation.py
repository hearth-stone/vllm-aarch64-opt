"""使用 vLLM 对 DeepSeek-V2-Lite 进行单次推理。

输出长度为 1 个 token，温度设置为 0（贪心采样），固定随机数种子以保证可复现性。

使用方式:
    python run_inference_once.py [--model MODEL_PATH] [--seed 42] [--prompt "你好"]

示例:
    # 使用默认模型路径
    python run_inference_once.py

    # 指定本地模型路径
    python run_inference_once.py --model /path/to/deepseek-v2-lite

    # 自定义 prompt
    python run_inference_once.py --prompt "Hello, my name is"

    # 启用 PyTorch Profiler，trace 文件中包含每个算子的 FLOPs 数据
    python run_inference_once.py --profile --profile-dir ./profiler_output
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import socket
import subprocess
import sys
import time
from datetime import datetime
from typing import TYPE_CHECKING

import numpy as np
import torch

from vllm import LLM, SamplingParams
from vllm.utils.argparse_utils import human_readable_int

if TYPE_CHECKING:
    from vllm.config import ProfilerConfig

# 默认模型名称（HuggingFace 上的路径或本地路径）
DEFAULT_MODEL = "deepseek-ai/DeepSeek-V2-Lite"
DEFAULT_SEED = 42
DEFAULT_PROMPT = "介绍下你自己:"
# Profiler 默认配置
DEFAULT_PROFILE_DIR = "./profiler_output"


def set_global_seed(seed: int) -> None:
    """设置全局随机数种子，确保可复现性。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def prompt_metadata(prompt: str) -> dict[str, object]:
    data = prompt.encode("utf-8", errors="surrogatepass")
    return {
        "chars": len(prompt),
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "preview": prompt[:128],
    }


def env_snapshot() -> dict[str, str]:
    keys = (
        "LD_PRELOAD",
        "MALLOC_CONF",
        "VLLM_CPU_KVCACHE_SPACE",
        "VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS",
        "VLLM_TORCH_PROFILER_SKIP_TRACE",
        "VLLM_TORCH_PROFILER_TABLE_ROW_LIMIT",
        "VLLM_DSV4_PREFILL_SHAPE_LOG",
        "VLLM_DSV4_PREFILL_SHAPE_LOG_RANKS",
        "VLLM_DSV4_PREFILL_SHAPE_LOG_LIMIT",
        "VLLM_CPU_MOE_ROUTING_DUMP_DIR",
        "VLLM_CPU_MOE_ROUTING_DUMP_LIMIT",
        "VLLM_CPU_ALLREDUCE_TIMING_DUMP_DIR",
        "VLLM_CPU_ALLREDUCE_TIMING_DUMP_LIMIT",
        "VLLM_CPU_ATTENTION_TIMING_DUMP_DIR",
        "VLLM_CPU_ATTENTION_TIMING_DUMP_LIMIT",
        "OMP_NUM_THREADS",
        "KMP_AFFINITY",
        "GOMP_CPU_AFFINITY",
        "GLOO_DEVICE_TRANSPORT",
        "GLOO_SOCKET_IFNAME",
    )
    return {key: os.environ[key] for key in keys if key in os.environ}


def git_snapshot() -> dict[str, object]:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    def git_output(*args: str) -> str | None:
        try:
            result = subprocess.run(
                ("git", *args),
                cwd=repo_root,
                check=True,
                capture_output=True,
                text=True,
                timeout=2,
            )
        except Exception:
            return None
        return result.stdout.strip()

    status = git_output("status", "--porcelain", "--untracked-files=no")
    return {
        "repo_root": repo_root,
        "commit": git_output("rev-parse", "HEAD"),
        "branch": git_output("branch", "--show-current"),
        "dirty_tracked": bool(status),
    }


def sanitized_args(args: argparse.Namespace) -> dict[str, object]:
    result = vars(args).copy()
    result["prompt"] = prompt_metadata(args.prompt)
    return result


def write_profile_manifest(
    path: str,
    *,
    args: argparse.Namespace,
    status: str,
    timings: dict[str, float],
    outputs: list[object] | None = None,
    error: str | None = None,
) -> None:
    generated_tokens = None
    prompt_tokens = None
    if outputs:
        first = outputs[0]
        prompt_token_ids = getattr(first, "prompt_token_ids", None)
        if prompt_token_ids is not None:
            prompt_tokens = len(prompt_token_ids)
        if getattr(first, "outputs", None):
            generated_tokens = len(first.outputs[0].token_ids)

    payload = {
        "schema_version": 1,
        "created_at": now_iso(),
        "status": status,
        "script": os.path.abspath(__file__),
        "cwd": os.getcwd(),
        "argv": sys.argv,
        "hostname": socket.gethostname(),
        "platform": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python": sys.version,
            "torch": torch.__version__,
        },
        "git": git_snapshot(),
        "args": sanitized_args(args),
        "env": env_snapshot(),
        "timings_s": timings,
        "metrics": {
            "prompt_tokens": prompt_tokens,
            "generated_tokens": generated_tokens,
        },
        "error": error,
    }

    manifest_path = os.path.abspath(path)
    parent = os.path.dirname(manifest_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    tmp_path = f"{manifest_path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp_path, manifest_path)


def append_request_timing(
    path: str,
    *,
    request_index: int,
    start_ns: int,
    end_ns: int,
    prompt: str,
    generated_tokens: int | None,
) -> None:
    payload = {
        "request_index": request_index,
        "start_ns": start_ns,
        "end_ns": end_ns,
        "duration_s": (end_ns - start_ns) / 1e9,
        "prompt": prompt_metadata(prompt),
        "generated_tokens": generated_tokens,
    }
    timing_path = os.path.abspath(path)
    parent = os.path.dirname(timing_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(timing_path, "a", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, sort_keys=True)
        f.write("\n")


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="使用 vLLM 对 DeepSeek-V2-Lite 进行单次推理",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"模型路径或 HuggingFace 模型名称 (默认: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"随机数种子 (默认: {DEFAULT_SEED})",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=DEFAULT_PROMPT,
        help="输入 prompt",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=2100,
        help="模型最大上下文长度 (默认: 4096)",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="张量并行大小 (默认: 1)",
    )
    parser.add_argument(
        "--load-format",
        type=str,
        default="auto",
        help="模型加载格式，例如 auto 或 sharded_state (默认: auto)",
    )
    parser.add_argument(
        "--enforce-eager",
        action="store_true",
        default=True,
        help="禁用 CUDA graph，强制使用 eager 模式 (适用于调试或非 CUDA 环境)",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        default=True,
        help="信任远程代码 (默认: True)",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=256,
        help="KV cache 块大小 (默认: 32，可选: 8, 16, 32, 64, 128)",
    )
    parser.add_argument(
        "--kv-cache-memory-bytes",
        type=human_readable_int,
        default=None,
        help="每个 TP worker 的 KV cache 大小，例如 2G/5G。默认交给 vLLM/env 决定。",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.4,
        help="未显式设置 KV cache 时的 CPU 内存利用率 (默认: 0.4)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="最大生成 token 数。profile decode 时默认 2，其它情况默认 1。",
    )
    parser.add_argument(
        "--repeat-requests",
        type=int,
        default=1,
        help="连续执行同一个 prompt 的次数；用于跳过首轮 cold-start。",
    )
    parser.add_argument(
        "--request-timing-log",
        type=str,
        default=None,
        help="写入每次 generate 的 perf_counter_ns 时间窗口 JSONL。",
    )
    parser.add_argument(
        "--enable-expert-parallel",
        "-ep",
        action="store_true",
        default=False,
        help="启用专家并行 (默认: False)",
    )
    parser.add_argument(
        "--no-expert-parallel",
        action="store_false",
        dest="enable_expert_parallel",
        help="禁用专家并行",
    )
    # ---- PyTorch Profiler 相关参数 ----
    parser.add_argument(
        "--profile",
        action="store_true",
        default=False,
        help="启用 PyTorch Profiler (默认: False)",
    )
    parser.add_argument(
        "--profile-baseline",
        action="store_true",
        default=False,
        help=(
            "启用固定 prefill profile 基线：prefill/max_tokens=1/"
            "record_shapes/no_trace/table_row_limit=1000。"
        ),
    )
    parser.add_argument(
        "--profile-dir",
        type=str,
        default=DEFAULT_PROFILE_DIR,
        help=f"Profiler 输出目录 (默认: {DEFAULT_PROFILE_DIR})",
    )
    parser.add_argument(
        "--profile-phase",
        choices=("decode", "prefill"),
        default="prefill",
        help="采样的阶段。默认 prefill；decode 会跳过长 prompt prefill。",
    )
    parser.add_argument(
        "--profile-delay-iterations",
        type=int,
        default=None,
        help="覆盖 profiler delay_iterations。默认 decode=2，prefill=0。",
    )
    parser.add_argument(
        "--profile-max-iterations",
        type=int,
        default=1,
        help="最多采样的 worker iteration 数 (默认: 1)",
    )
    parser.add_argument(
        "--profile-with-flops",
        action="store_true",
        default=False,
        help="启用 PyTorch profiler FLOPs 统计。长 prompt 上容易放大内存。",
    )
    parser.add_argument(
        "--profile-no-trace",
        action="store_true",
        default=False,
        help="只写 profiler_out_*.txt，不导出 chrome trace，降低 stop/export 内存。",
    )
    parser.add_argument(
        "--profile-table-row-limit",
        type=int,
        default=None,
        help="record_shapes=True 时 profiler_out 表最多输出多少行；默认 1000。",
    )
    parser.add_argument(
        "--profile-record-shapes",
        action="store_true",
        dest="profile_record_shapes",
        default=None,
        help="启用 torch profiler record_shapes。prefill 上容易 OOM。",
    )
    parser.add_argument(
        "--profile-no-record-shapes",
        action="store_false",
        dest="profile_record_shapes",
        help="关闭 torch profiler record_shapes。",
    )
    parser.add_argument(
        "--prefill-shape-log",
        type=str,
        default=None,
        help=(
            "启用 DeepSeek-V4 prefill 轻量 shape JSONL 日志，"
            "建议用于 prefill FLOPs 分析。"
        ),
    )
    parser.add_argument(
        "--prefill-shape-log-ranks",
        type=str,
        default="0",
        help="shape 日志记录的 TP rank，默认只记录 0；可设为 all 或 0,1。",
    )
    parser.add_argument(
        "--prefill-shape-log-limit",
        type=int,
        default=0,
        help="shape 日志最大记录数，0 表示不限制。",
    )
    parser.add_argument(
        "--profile-manifest",
        type=str,
        default=None,
        help="写入本次 profile 的 JSON manifest，记录命令、环境、耗时和 token 数。",
    )
    return parser.parse_args()


def apply_profile_baseline_defaults(args: argparse.Namespace) -> None:
    if not args.profile_baseline:
        return
    args.profile = True
    args.profile_phase = "prefill"
    args.profile_no_trace = True
    args.profile_max_iterations = 1
    args.max_tokens = 1 if args.max_tokens is None else args.max_tokens
    if args.profile_record_shapes is None:
        args.profile_record_shapes = True
    if args.profile_table_row_limit is None:
        args.profile_table_row_limit = 1000


def build_profiler_config(
    profile_dir: str,
    *,
    phase: str,
    delay_iterations: int | None,
    max_iterations: int,
    record_shapes: bool,
    with_flops: bool,
) -> ProfilerConfig:
    """构建 vLLM ProfilerConfig，用于在 worker 进程内部启动 Torch Profiler。

    vLLM 的实际计算发生在 worker 子进程中，必须通过 ``profiler_config``
    参数传入 ``LLM(...)``，worker 才会在初始化时创建 Profiler 实例。
    直接使用环境变量或 ``llm.start_profile()`` 在未预先配置的情况下会
    抛出 ``RuntimeError: Profiler is not enabled.``。

    **重要：内存开销控制**
    对于长 prompt（耗时数千秒），同时开启 ``with_stack`` + ``with_memory`` +
    ``record_shapes`` 会导致 profiler 在 ``stop`` 时累积数百 GB 事件数据，
    引发 worker 进程被 OOM killer 杀掉。因此：

    1. 关闭 ``with_stack`` / ``with_memory``（二者是主要内存放大因子）；
    2. 通过 ``delay_iterations`` + ``max_iterations`` 只 profile 少量
       engine iteration（decode 过程每一步的 kernel 分布几乎一致，
       采样几步即可代表整体热点）；
    3. ``ignore_frontend=True`` 关闭 AsyncLLM 前端 profiling，避免覆盖
       整条 prompt 时间轴造成额外开销。

    :param profile_dir: trace 文件输出目录（必须为绝对路径）。
    :param phase: 要采样的阶段，decode 默认跳过首个 prefill iteration。
    :param delay_iterations: 显式覆盖 delay_iterations。
    :param max_iterations: 最大采样 worker iteration 数。
    :param record_shapes: 是否记录输入形状。
    :param with_flops: 是否启用 PyTorch FLOPs 统计。
    :return: 配置好的 ProfilerConfig 实例。
    """
    from vllm.config import ProfilerConfig

    abs_dir = os.path.abspath(profile_dir)
    os.makedirs(abs_dir, exist_ok=True)

    if delay_iterations is None:
        # WorkerProfiler.step() 在执行当前 worker iteration 之前调用。
        # delay=2 会让第一步 prefill 不被记录，第二步 decode 开始采样。
        delay_iterations = 2 if phase == "decode" else 0

    return ProfilerConfig(
        profiler="torch",
        torch_profiler_dir=abs_dir,
        # --- 内存开销控制：关掉两个“吃内存大户” ---
        torch_profiler_with_stack=False,
        torch_profiler_with_memory=False,
        # --- 形状/FLOPs 统计：默认只在 decode 上打开 shapes ---
        torch_profiler_record_shapes=record_shapes,
        torch_profiler_with_flops=with_flops,
        # Long CPU prefill should normally run with trace export disabled. Keep
        # gzip off for the fallback trace path so an interrupted export is less
        # likely to leave unusable .gz files.
        torch_profiler_use_gzip=False,
        # CPU worker 只有 CPU activity，按 CUDA 时间排序没有意义，还会增加
        # stop/export 阶段的 key_averages 开销。
        torch_profiler_dump_cuda_time_total=False,
        # --- 窗口化：只采样极少 iteration，避免积累数 TB trace ---
        ignore_frontend=True,
        delay_iterations=delay_iterations,
        max_iterations=max_iterations,
        warmup_iterations=0,
        active_iterations=1,
        wait_iterations=0,
    )


def main() -> None:
    """主函数：执行单次推理。"""
    script_start = time.perf_counter()
    args = parse_args()
    apply_profile_baseline_defaults(args)
    if args.repeat_requests <= 0:
        raise ValueError("--repeat-requests must be positive")
    timings: dict[str, float] = {}
    outputs = []
    status = "ok"
    error = None

    # 1. 固定全局随机数种子
    set_global_seed(args.seed)
    print(f"随机数种子: {args.seed}")
    print(f"模型: {args.model}")
    print(f"Prompt: {args.prompt!r}")
    if args.profile:
        print(f"Profiler 已启用，输出目录: {os.path.abspath(args.profile_dir)}")
    if args.profile and args.profile_phase == "prefill" and not args.prefill_shape_log:
        args.prefill_shape_log = os.path.join(
            args.profile_dir,
            "prefill_shapes_rank{rank}.jsonl",
        )
    if args.profile_no_trace:
        os.environ["VLLM_TORCH_PROFILER_SKIP_TRACE"] = "1"
        print("Profiler chrome trace 导出已关闭，只保留 profiler_out_*.txt")
    if args.profile_table_row_limit is not None:
        os.environ["VLLM_TORCH_PROFILER_TABLE_ROW_LIMIT"] = str(
            args.profile_table_row_limit
        )
    if args.prefill_shape_log:
        prefill_shape_log = os.path.abspath(args.prefill_shape_log)
        os.environ["VLLM_DSV4_PREFILL_SHAPE_LOG"] = prefill_shape_log
        os.environ["VLLM_DSV4_PREFILL_SHAPE_LOG_RANKS"] = (
            args.prefill_shape_log_ranks
        )
        os.environ["VLLM_DSV4_PREFILL_SHAPE_LOG_LIMIT"] = str(
            args.prefill_shape_log_limit
        )
        parent = os.path.dirname(prefill_shape_log)
        if parent:
            os.makedirs(parent, exist_ok=True)
        print(f"Prefill shape log: {prefill_shape_log}")
    print("-" * 60)

    try:
        # 2. 初始化 vLLM 引擎
        # seed 参数会传递给 vLLM 引擎内部，确保模型初始化和推理的可复现性
        # 当启用 profile 时，通过 profiler_config 让 worker 内部创建 Profiler，
        # 默认只记录 decode 的输入形状，prefill 的长 CPU trace 很容易在 stop/export
        # 阶段 OOM。
        profile_record_shapes = args.profile_record_shapes
        if profile_record_shapes is None:
            # prefill 的 CPU trace 很长，默认不要让 torch profiler 保存每个
            # ATen op 的输入形状；需要形状时用 --prefill-shape-log。
            profile_record_shapes = args.profile_phase == "decode"
        if args.profile and args.profile_phase == "prefill" and profile_record_shapes:
            print(
                "[Warning] prefill + torch_profiler_record_shapes=True 会显著拖慢 "
                "profiler stop/export；建议保持 --profile-no-trace，且不要打开 "
                "--profile-with-flops。"
            )

        profiler_config = (
            build_profiler_config(
                args.profile_dir,
                phase=args.profile_phase,
                delay_iterations=args.profile_delay_iterations,
                max_iterations=args.profile_max_iterations,
                record_shapes=profile_record_shapes,
                with_flops=args.profile_with_flops,
            )
            if args.profile
            else None
        )
        llm_kwargs = dict(
            model=args.model,
            seed=args.seed,
            max_model_len=args.max_model_len,
            tensor_parallel_size=args.tensor_parallel_size,
            load_format=args.load_format,
            enforce_eager=args.enforce_eager,
            trust_remote_code=args.trust_remote_code,
            generation_config="vllm",
            profiler_config=profiler_config,
            enable_chunked_prefill=False,
            block_size=args.block_size,
            enable_prefix_caching=False,
            enable_expert_parallel=args.enable_expert_parallel,
            gpu_memory_utilization=args.gpu_memory_utilization,
        )
        if args.kv_cache_memory_bytes is not None:
            llm_kwargs["kv_cache_memory_bytes"] = args.kv_cache_memory_bytes
        init_start = time.perf_counter()
        llm = LLM(**llm_kwargs)
        timings["engine_init"] = time.perf_counter() - init_start

        # 3. 配置采样参数
        # temperature=0 => 贪心采样（确定性输出）
        # max_tokens=1 => 只生成 1 个 token；profile decode 时至少需要 2 个 token
        # repetition_penalty=1.1 => 重复惩罚，降低已生成 token 的概率
        # seed => 采样级别的随机种子（双重保险）
        max_tokens = args.max_tokens
        if max_tokens is None:
            max_tokens = 2 if args.profile and args.profile_phase == "decode" else 1
        if args.profile and args.profile_phase == "decode" and max_tokens < 2:
            raise ValueError("profile decode 至少需要 --max-tokens 2")

        sampling_params = SamplingParams(
            temperature=0,
            max_tokens=max_tokens,
            repetition_penalty=1.1,
            seed=args.seed,
        )

        # 4. 执行推理
        if args.profile:
            profile_start = time.perf_counter()
            llm.start_profile()
            timings["profile_start"] = time.perf_counter() - profile_start

        generate_start = time.perf_counter()
        for request_index in range(args.repeat_requests):
            request_start_ns = time.perf_counter_ns()
            outputs = llm.generate([args.prompt], sampling_params)
            request_end_ns = time.perf_counter_ns()
            generated_tokens = None
            if outputs and getattr(outputs[0], "outputs", None):
                generated_tokens = len(outputs[0].outputs[0].token_ids)
            if args.request_timing_log:
                append_request_timing(
                    args.request_timing_log,
                    request_index=request_index,
                    start_ns=request_start_ns,
                    end_ns=request_end_ns,
                    prompt=args.prompt,
                    generated_tokens=generated_tokens,
                )
            print(
                f"[Request Timing] index={request_index} "
                f"duration_s={(request_end_ns - request_start_ns) / 1e9:.3f}"
            )
        timings["generate"] = time.perf_counter() - generate_start

        if args.profile:
            stop_start = time.perf_counter()
            llm.stop_profile()
            timings["profile_stop"] = time.perf_counter() - stop_start
            print(f"[Profiler] 输出已保存到: {os.path.abspath(args.profile_dir)}")

        # 5. 输出结果
        for output in outputs:
            prompt = output.prompt
            generated_text = output.outputs[0].text
            token_ids = output.outputs[0].token_ids
            print(f"Prompt:          {prompt!r}")
            print(f"Generated text:  {generated_text!r}")
            print(f"Token IDs:       {token_ids}")
            print(f"Num tokens:      {len(token_ids)}")
    except BaseException as exc:
        status = "failed"
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        timings["total"] = time.perf_counter() - script_start
        if args.profile_manifest:
            write_profile_manifest(
                args.profile_manifest,
                args=args,
                status=status,
                timings=timings,
                outputs=outputs,
                error=error,
            )
            print(
                "[Profile Manifest] 已保存到: "
                f"{os.path.abspath(args.profile_manifest)}"
            )

    print("-" * 60)
    print("推理完成。")


if __name__ == "__main__":
    main()
