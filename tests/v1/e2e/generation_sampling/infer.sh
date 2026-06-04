#!/usr/bin/env bash
set -euo pipefail

export LD_PRELOAD=/usr/lib64/libjemalloc.so
export MALLOC_CONF="thp:always,oversize_threshold:2097152,background_thread:true"
export GLOO_DEVICE_TRANSPORT="TCP"
export GLOO_SOCKET_IFNAME="eno1"
export VLLM_CPU_KVCACHE_SPACE=5
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
PYTHON="${REPO_ROOT}/.venv/bin/python"
MODEL="${MODEL:-/mnt/models/DeepSeek-V4-Flash-BF16}"
TP="${TP:-4}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-2100}"
BLOCK_SIZE="${BLOCK_SIZE:-256}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.4}"
PROFILE_ROOT="${PROFILE_ROOT:-${SCRIPT_DIR}/profiler_output/deepseek_v4_flash_bf16}"
PROFILE_STAMP="${PROFILE_STAMP:-$(date +%Y%m%d-%H%M%S)}"
PROFILE_DIR="${PROFILE_DIR:-${PROFILE_ROOT}/${PROFILE_STAMP}}"
PROFILE_TABLE_ROW_LIMIT="${PROFILE_TABLE_ROW_LIMIT:-1000}"
PROFILE_LIMIT="${PROFILE_LIMIT:-80}"
GEMM_SHAPE_LIMIT="${GEMM_SHAPE_LIMIT:-200}"
PEAK_GFLOPS_PER_CORE="${PEAK_GFLOPS_PER_CORE:-92}"
THREADS_PER_RANK="${THREADS_PER_RANK:-64}"
CORES_PER_NUMA="${CORES_PER_NUMA:-80}"
CORES_PER_RANK="${CORES_PER_RANK:-${THREADS_PER_RANK}}"
MANIFEST="${PROFILE_DIR}/run_manifest.json"
SUMMARY="${PROFILE_DIR}/summary.txt"

#export VLLM_CPU_AWQ_USE_FUSED_CPP=1
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=36000
if [[ -z "${VLLM_CPU_OMP_THREADS_BIND:-}" && -z "${VLLM_CPU_NUM_OF_RESERVED_CPU:-}" ]]; then
    export VLLM_CPU_NUM_OF_RESERVED_CPU="$(((CORES_PER_NUMA - THREADS_PER_RANK) * TP))"
fi
# python run_inference_long.py --tensor-parallel-size 4 --model /mnt/models/DeepSeek-V2-Lite/ --profile --profile-dir ./profiler_output/deepseek_v2_lite
# python run_inference_long.py --tensor-parallel-size 4 --model /mnt/models/DeepSeek-V2-Lite/
# python run_inference_long.py --tensor-parallel-size 4 --model /mnt/models/DeepSeek-V2/
# python run_inference_once.py --tensor-parallel-size 4 --model /mnt/models/DeepSeek-V2/
# python run_inference_long.py --tensor-parallel-size 4 --model /mnt/models/DeepSeek-R1-AWQ --profile --profile-dir ./profiler_output/deepseek_r1_awq
# python run_inference_once.py --tensor-parallel-size 1 --model  /mnt/models/Qwen2.5-7B-Instruct-AWQ --profile --profile-dir ./profiler_output/qwen2.5_7b_awq
# python run_inference_once.py --tensor-parallel-size 4  --model  /mnt/models/DeepSeek-V4-Flash-BF16
# python run_inference_long.py --tensor-parallel-size 4 --model /mnt/models/DeepSeek-V2/ --profile --profile-dir ./profiler_output/deepseek_v2
mkdir -p "${PROFILE_DIR}"

extra_args=()
if [[ -n "${KV_CACHE_MEMORY_BYTES:-}" ]]; then
    extra_args+=(--kv-cache-memory-bytes "${KV_CACHE_MEMORY_BYTES}")
fi

echo "[baseline] profile_dir=${PROFILE_DIR}"
echo "[baseline] manifest=${MANIFEST}"
echo "[baseline] summary=${SUMMARY}"
echo "[baseline] model=${MODEL}"
echo "[baseline] threads_per_rank=${THREADS_PER_RANK}"
echo "[baseline] cores_per_rank=${CORES_PER_RANK}"
echo "[baseline] vllm_reserved_cpus=${VLLM_CPU_NUM_OF_RESERVED_CPU:-unset}"
echo "[baseline] omp_threads_bind=${VLLM_CPU_OMP_THREADS_BIND:-auto}"

"${PYTHON}" "${SCRIPT_DIR}/generation.py" \
    --tensor-parallel-size "${TP}" \
    --model "${MODEL}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --block-size "${BLOCK_SIZE}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    --profile-baseline \
    --profile-dir "${PROFILE_DIR}" \
    --profile-table-row-limit "${PROFILE_TABLE_ROW_LIMIT}" \
    --profile-manifest "${MANIFEST}" \
    "${extra_args[@]}"

"${PYTHON}" "${SCRIPT_DIR}/profiler_output/prof.py" "${PROFILE_DIR}" \
    --limit "${PROFILE_LIMIT}" \
    --gemm-shape-limit "${GEMM_SHAPE_LIMIT}" \
    --peak-gflops-per-core "${PEAK_GFLOPS_PER_CORE}" \
    --cores-per-rank "${CORES_PER_RANK}" \
    > "${SUMMARY}"

cat "${SUMMARY}"
