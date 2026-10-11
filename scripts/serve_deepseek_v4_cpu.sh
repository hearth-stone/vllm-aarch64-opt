#!/usr/bin/env bash
# TP4，每个 rank 80 核，使用 LLVM OpenMP，PyTorch 保留 mimalloc，普通分配使用 jemalloc。
set -euo pipefail

# 通过注释切换模型，下面三行只保留一行启用。
model=/mnt/models/DeepSeek-V4-Flash-BF16
# model=/mnt/models/DeepSeek-V4-Flash-INT8
# model=/mnt/models/DeepSeek-V4-Pro-Base

# 最大上下文长度，同时作为每个调度批次的总 token 预算。
export MAX_MODEL_LEN=9216
# 每个 TP rank 的 KV cache 容量，单位 GiB；不是所有 rank 的总容量。
export KV_GIB_PER_RANK=32
kv_bytes=$((KV_GIB_PER_RANK * 1024 * 1024 * 1024))

# 必要环境: vllm目录, fused_cpp目录和python
# vllm_root=$(cd "$(dirname "$0")/.." && pwd)
vllm_root=/home/zhangxu/final_test/vllm-aarch64-opt
source /home/zhangxu/final_test/test/bin/activate
fused_root=/home/zhangxu/final_test/fused_cpp

run_dir="$vllm_root/logs/$(basename "$model")-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$run_dir"

# 使用仓库内的固定校准表，各 rank 采用相同的 rank0 校准数值并保留各自绑核信息。
export FUSED_CPP_MOE_PLANNER_PROFILE="$vllm_root/benchmarks/profiles/deepseek_v4/arm-codex-tp4-80cores/rank{tp_rank}.json"
# 已有预打包缓存仅对应 Flash BF16，其他模型仍在启动时打包。
if [[ "$model" == /mnt/models/DeepSeek-V4-Flash-BF16 ]]; then
    # 直接加载 Flash BF16 已打包的专家权重，避免重新加载和打包原始专家权重。
    export VLLM_CPU_MOE_PREPACKED_DIR=/mnt/models/.cache/zhangxu/DeepSeek-V4-Flash-BF16-moe-prepacked-v028-482e03e
else
    # INT8 和 Pro 不复用 BF16 缓存。
    unset VLLM_CPU_MOE_PREPACKED_DIR
fi
# 加载预打包缓存时提前触碰每个内存页，避免首次推理才读取 mmap 权重页。
export VLLM_CPU_MOE_PREPACK_PREFAULT=1
# 清除旧版 KV 容量设置，由 --kv-cache-memory-bytes 指定容量。
unset VLLM_CPU_KVCACHE_SPACE

# 加载当前 vLLM 代码和 fused_cpp，保留 PyTorch 默认 CPU 分配器。
export PYTHONPATH="$fused_root/src:$vllm_root"
# 加载 LLVM OpenMP，并让普通 malloc/free 使用 jemalloc；PyTorch 张量仍使用内置 mimalloc。
export LD_PRELOAD=/usr/local/lib/libomp.so:/usr/lib64/libjemalloc.so
# 禁用 PyTorch 内置 mimalloc 的空闲页 purge，供后续张量分配复用。
export MIMALLOC_PURGE_DELAY=-1
# 保留 jemalloc 的可复用缓冲区，启用后台维护，并允许分配器元数据使用透明大页。
export MALLOC_CONF=background_thread:true,metadata_thp:auto,oversize_threshold:0,dirty_decay_ms:-1,muzzy_decay_ms:-1
# 明确选择 CPU 后端。
export VLLM_TARGET_DEVICE=cpu
# 要求使用 fused_cpp 的 DeepSeek V4 CPU 内核，不支持的路径直接报错。
export VLLM_CPU_FUSED_CPP_STRICT=1
# 为四个 TP rank 分别绑定 80 个 CPU 核，满足 Plan V2 的显式绑核要求。
export VLLM_CPU_OMP_THREADS_BIND='0-79|80-159|160-239|240-319'
# 启动时用 16 个线程并行打包专家权重，避免 Pro 默认单线程打包耗时过长。
export FUSED_CPP_MOE_PREPACK_THREADS=16
# 指定本机 eno1 网卡，用于 Gloo 的 TP 通信。
export GLOO_SOCKET_IFNAME=eno1

# 原始 FP8 模型在加载时转换为 CPU W8A8；覆盖配置只写入日志目录。
python - "$model" "$run_dir/hf-overrides.json" <<'PYCONFIG'
import json
import sys
from pathlib import Path

config = json.loads((Path(sys.argv[1]) / "config.json").read_text())
quant = config.get("quantization_config") or {}
overrides = {}
if config.get("model_type") == "deepseek_v4" and quant.get("quant_method") == "fp8":
    from vllm.models.deepseek_v4.cpu.fp8_requant import make_cpu_w8a8_quantization_config

    overrides = {
        "cpu_fp8_to_int8": True,
        "cpu_fp8_source_block_size": quant["weight_block_size"],
        "cpu_fp8_conversion_rows_per_chunk": 2048,
        "quantization_config": make_cpu_w8a8_quantization_config(config["num_hidden_layers"]),
    }
Path(sys.argv[2]).write_text(json.dumps(overrides))
PYCONFIG

echo "Model: $model; service: http://127.0.0.1:8018; logs: $run_dir/server.log"
echo "$$" > "$run_dir/server.pid"
printf '%s\n' "$model" > "$run_dir/model-path.txt"
exec python -m vllm.entrypoints.openai.api_server \
    --model "$model" --served-model-name deepseek-v4 \
    --hf-overrides "$(cat "$run_dir/hf-overrides.json")" \
    --host 127.0.0.1 --port 8018 --tensor-parallel-size 4 --dtype bfloat16 \
    --max-model-len "$MAX_MODEL_LEN" --max-num-batched-tokens "$MAX_MODEL_LEN" \
    --block-size 256 --kv-cache-memory-bytes "$kv_bytes" \
    --trust-remote-code --enforce-eager \
    --no-enable-chunked-prefill --no-enable-prefix-caching \
    --numa-bind --numa-bind-node-groups 0 1 2 3 \
    --numa-bind-cpus 0-79 80-159 160-239 240-319 \
    > "$run_dir/server.log" 2>&1
