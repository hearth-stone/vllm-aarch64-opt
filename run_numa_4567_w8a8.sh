#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# DeepSeek-V4-Flash-INT8 W8A8 on Arm-codex-internal NUMA nodes 4, 5, 6, 7.
# Same CPU bind as run_numa_4567.sh. Adds compressed-tensors quantization.
#
# Node 4: 160-199
# Node 5: 200-239
# Node 6: 240-279
# Node 7: 280-319
#
# Without an explicit planner profile, each rank calibrates on its bound CPUs
# during MoE initialization.

set -euo pipefail

ROOT="/home/zhangxu/codex/vllm-aarch64-v0.28.0"
expected_nodes=(4 5 6 7)
expected_cpus=("160-199" "200-239" "240-279" "280-319")
for i in "${!expected_nodes[@]}"; do
    node="${expected_nodes[$i]}"
    want="${expected_cpus[$i]}"
    cpulist_path="/sys/devices/system/node/node${node}/cpulist"
    if [[ ! -f "$cpulist_path" ]]; then
        echo "missing $cpulist_path" >&2
        exit 1
    fi
    got="$(tr -d '[:space:]' < "$cpulist_path")"
    if [[ "$got" != "$want" ]]; then
        echo "NUMA node $node cpulist is '$got', expected '$want'" >&2
        exit 1
    fi
done

export LD_PRELOAD=/usr/lib64/libjemalloc.so
export MALLOC_CONF="thp:always,oversize_threshold:2097152,background_thread:true"
export GLOO_DEVICE_TRANSPORT="TCP"
export GLOO_SOCKET_IFNAME="eno1"
export VLLM_CPU_KVCACHE_SPACE=10
export VLLM_TARGET_DEVICE=cpu
export VLLM_CPU_FUSED_CPP_STRICT=1
export CPU_LIMIT_PER_WORKER=40
export TP_SIZE=4
export OMP_THREADS_PER_RANK=$CPU_LIMIT_PER_WORKER
export VLLM_CPU_OMP_THREADS_BIND="160-199|200-239|240-279|280-319"

export OMP_NUM_THREADS=$OMP_THREADS_PER_RANK
export MKL_NUM_THREADS=$OMP_THREADS_PER_RANK
export NUMEXPR_MAX_THREADS=$OMP_THREADS_PER_RANK
export OPENBLAS_NUM_THREADS=$OMP_THREADS_PER_RANK
export VECLIB_MAXIMUM_THREADS=$OMP_THREADS_PER_RANK
export GOTO_NUM_THREADS=$OMP_THREADS_PER_RANK

export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=36000

exec "$ROOT/.venv/bin/python" -m vllm.entrypoints.openai.api_server \
    --model "/mnt/models/DeepSeek-V4-Flash-INT8/" \
    --host "127.0.0.1" \
    --port "8004" \
    --gpu-memory-utilization "0.95" \
    --max-model-len "4096" \
    --tensor-parallel-size "$TP_SIZE" \
    --dtype "bfloat16" \
    --trust-remote-code \
    --quantization compressed-tensors \
    --numa-bind \
    --numa-bind-nodes 4 5 6 7 \
    --numa-bind-cpus 160-199 200-239 240-279 280-319 \
    "$@"
