#!/usr/bin/env bash
# DeepSeek V4 Flash W8A8 INT8 precision run: TP4, one 40-core NUMA node per rank.
set -euo pipefail

vllm_root="${VLLM_ROOT:-/home/zhangxu/codex/vllm-aarch64-v0.28.0}"
vllm_python="${VLLM_PYTHON:-/home/zhangxu/codex/vllm-aarch64-v0.28.0/.venv/bin/python}"
fused_cpp_src="${FUSED_CPP_SRC:-/home/zhangxu/codex/fused_cpp-dsv4-v028-delivery/src}"

for node in 4 5 6 7; do
    cpulist_path="/sys/devices/system/node/node${node}/cpulist"
    expected="$((node * 40))-$((node * 40 + 39))"
    [[ -f "$cpulist_path" ]] || { echo "Missing $cpulist_path" >&2; exit 1; }
    actual="$(tr -d '[:space:]' < "$cpulist_path")"
    [[ "$actual" == "$expected" ]] || {
        echo "NUMA node $node has CPUs $actual; expected $expected" >&2
        exit 1
    }
done

[[ -x "$vllm_python" && -d "$vllm_root/vllm" && -d "$fused_cpp_src/fused_cpp" ]] || {
    echo "VLLM_PYTHON, VLLM_ROOT, or FUSED_CPP_SRC is unavailable" >&2
    exit 1
}

unset FUSED_CPP_MOE_PLANNER_PROFILE
export PYTHONPATH="$fused_cpp_src:$vllm_root${PYTHONPATH:+:$PYTHONPATH}"
export LD_PRELOAD=/usr/lib64/libjemalloc.so
export MALLOC_CONF=thp:always,oversize_threshold:2097152,background_thread:true
export GLOO_DEVICE_TRANSPORT=TCP
export GLOO_SOCKET_IFNAME=eno1
export VLLM_CPU_KVCACHE_SPACE=10
export VLLM_TARGET_DEVICE=cpu
export VLLM_CPU_FUSED_CPP_STRICT=1
export VLLM_CPU_OMP_THREADS_BIND='160-199|200-239|240-279|280-319'
export OMP_NUM_THREADS=40
export MKL_NUM_THREADS=40
export NUMEXPR_MAX_THREADS=40
export OPENBLAS_NUM_THREADS=40
export VECLIB_MAXIMUM_THREADS=40
export GOTO_NUM_THREADS=40
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=36000

cd "$vllm_root"
exec "$vllm_python" -m vllm.entrypoints.openai.api_server \
    --model /mnt/models/DeepSeek-V4-Flash-INT8/ \
    --host 127.0.0.1 --port 8004 \
    --gpu-memory-utilization 0.95 --max-model-len 4096 \
    --tensor-parallel-size 4 --dtype bfloat16 --trust-remote-code \
    --quantization compressed-tensors \
    --numa-bind --numa-bind-nodes 4 5 6 7 \
    --numa-bind-cpus 160-199 200-239 240-279 280-319 \
    "$@"
