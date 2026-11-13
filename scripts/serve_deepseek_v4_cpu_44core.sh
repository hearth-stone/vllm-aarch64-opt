#!/usr/bin/env bash
# Arm2：TP4，每个 rank 44 核（每个 NUMA 节点 48 核，空出 44-47 等核给 SSP 等后台服务）。
# 2048 token 输入、1 token 输出实测：W8A8 约 3.5 秒，BF16 约 6.6 秒。
# SSP 需先限制到空出的核上（重启后失效，需重新执行）：
#   sudo systemctl set-property --runtime ssp-daemon.service CPUWeight=100 AllowedCPUs=47,95,143,191
set -euo pipefail

# 通过注释切换模型，下面两行只保留一行启用。
model=/home/zhangxu/models/DeepSeek-V4-Flash-INT8
# model=/home/zhangxu/models/DeepSeek-V4-Flash-BF16

# 最大上下文长度，同时作为每个调度批次的总 token 预算。
export MAX_MODEL_LEN=9216
# 每个 TP rank 的 KV cache 容量，单位 GiB；不是所有 rank 的总容量。
export KV_GIB_PER_RANK=32
kv_bytes=$((KV_GIB_PER_RANK * 1024 * 1024 * 1024))

# 必要环境: vllm目录, fused_cpp目录和python；默认值为 Arm2 现有部署。
vllm_root=${VLLM_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}
fused_root=${FUSED_CPP_ROOT:-/home/zhangxu/dsv4-flash-perf-20261025/fused_cpp-dsv4-v028-aot-delivery}
python_bin=${VLLM_PYTHON:-/home/zhangxu/dsv4-flash-perf-20261025/.venv/bin/python}

run_dir="$vllm_root/logs/$(basename "$model")-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$run_dir"

# 不使用固定校准表，各 rank 启动时在自己绑定的 44 个核上自动校准。
unset FUSED_CPP_MOE_PLANNER_PROFILE
# 已有预打包缓存按模型区分；目录不存在时在启动时打包。
prepack_root=${PREPACK_ROOT:-/home/zhangxu/dsv4-prepack-cache.ZlWENHeYnF}
case "$model" in
    *INT8) prepack_dir="$prepack_root/w8a8" ;;
    *BF16) prepack_dir="$prepack_root/bf16" ;;
    *) prepack_dir= ;;
esac
if [[ -n "$prepack_dir" && -d "$prepack_dir" ]]; then
    # 直接加载已打包的专家权重，避免重新加载和打包原始专家权重。
    export VLLM_CPU_MOE_PREPACKED_DIR="$prepack_dir"
else
    unset VLLM_CPU_MOE_PREPACKED_DIR
fi
unset VLLM_CPU_MOE_PREPACK_EXPORT_DIR
# 加载预打包缓存时提前触碰每个内存页，避免首次推理才读取 mmap 权重页。
export VLLM_CPU_MOE_PREPACK_PREFAULT=1
# 清除旧版 KV 容量设置，由 --kv-cache-memory-bytes 指定容量。
unset VLLM_CPU_KVCACHE_SPACE
# 线程数和绑核由 vLLM 按 VLLM_CPU_OMP_THREADS_BIND 设置，清除可能冲突的 OpenMP/BLAS 设置。
unset OMP_NUM_THREADS OMP_THREAD_LIMIT OMP_PROC_BIND OMP_PLACES GOMP_CPU_AFFINITY KMP_AFFINITY
unset MKL_NUM_THREADS OPENBLAS_NUM_THREADS NUMEXPR_MAX_THREADS KMP_BLOCKTIME KMP_TPAUSE
unset KMP_USE_YIELD OMP_WAIT_POLICY KMP_LIBRARY KMP_SETTINGS KMP_HOT_TEAMS_MODE KMP_HOT_TEAMS_MAX_LEVEL
unset FUSED_CPP_MOE_PIN_THREADS FUSED_CPP_MOE_PIN_THREAD_CPUS FUSED_CPP_MOE_TRACE FUSED_CPP_MOE_TRACE_FILE

# 加载当前 vLLM 代码和 fused_cpp，保留 PyTorch 默认 CPU 分配器。
export PYTHONPATH="$fused_root/src:$fused_root:$vllm_root"
# 加载 LLVM OpenMP，并让普通 malloc/free 使用 jemalloc；PyTorch 张量仍使用内置 mimalloc。
export LD_PRELOAD=/usr/lib64/libomp.so:/usr/lib64/libjemalloc.so.2
# 禁用 PyTorch 内置 mimalloc 的空闲页 purge，供后续张量分配复用。
export MIMALLOC_PURGE_DELAY=-1
# 保留 jemalloc 的可复用缓冲区，启用后台维护，并允许分配器元数据使用透明大页。
export MALLOC_CONF=background_thread:true,metadata_thp:auto,oversize_threshold:0,dirty_decay_ms:-1,muzzy_decay_ms:-1
# 明确选择 CPU 后端。
export VLLM_TARGET_DEVICE=cpu
# 要求使用 fused_cpp 的 DeepSeek V4 CPU 内核，不支持的路径直接报错。
export VLLM_CPU_FUSED_CPP_STRICT=1
# 为四个 TP rank 分别绑定 44 个 CPU 核；每个 NUMA 节点空出最后 4 个核。
# W8A8 MoE 的组宽度需同时整除核数和 256，44 核可用宽度 1/2/4；46、47 核明显变慢。
export VLLM_CPU_OMP_THREADS_BIND='0-43|48-91|96-139|144-187'
export VLLM_CPU_NUM_OF_RESERVED_CPU=0
# 启动时用 16 个线程并行打包专家权重。
export FUSED_CPP_MOE_PREPACK_THREADS=16
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=36000
# 指定本机 enp166s0f0 网卡，用于 Gloo 的 TP 通信。
export GLOO_SOCKET_IFNAME=enp166s0f0 GLOO_DEVICE_TRANSPORT=TCP
# 模型和 tokenizer 只从本地读取。
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

echo "Model: $model; service: http://127.0.0.1:8018; logs: $run_dir/server.log"
echo "$$" > "$run_dir/server.pid"
printf '%s\n' "$model" > "$run_dir/model-path.txt"
# 以最高优先级运行；没有权限时 nice 给出警告并按默认优先级运行。
exec nice -n "${NICE_LEVEL:--20}" "$python_bin" -m vllm.entrypoints.openai.api_server \
    --model "$model" --served-model-name deepseek-v4 \
    --host 127.0.0.1 --port 8018 --tensor-parallel-size 4 --dtype bfloat16 \
    --max-model-len "$MAX_MODEL_LEN" --max-num-batched-tokens "$MAX_MODEL_LEN" \
    --block-size 256 --kv-cache-memory-bytes "$kv_bytes" \
    --safetensors-load-strategy lazy \
    --trust-remote-code --enforce-eager \
    --no-enable-chunked-prefill --no-enable-prefix-caching \
    --numa-bind --numa-bind-node-groups 0 1 2 3 \
    --numa-bind-cpus 0-43 48-91 96-139 144-187 \
    > "$run_dir/server.log" 2>&1
