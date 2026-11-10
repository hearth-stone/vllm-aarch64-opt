export LD_PRELOAD=/usr/lib64/libjemalloc.so
export MALLOC_CONF="thp:always,oversize_threshold:2097152,background_thread:true"
export GLOO_DEVICE_TRANSPORT="TCP"
export GLOO_SOCKET_IFNAME="eno1"
export VLLM_CPU_KVCACHE_SPACE=10
export VLLM_TARGET_DEVICE=cpu
export VLLM_CPU_FUSED_CPP_STRICT=1
export CPU_LIMIT_PER_WORKER=80
export TP_SIZE=4
export OMP_THREADS_PER_RANK=$CPU_LIMIT_PER_WORKER
export VLLM_CPU_OMP_THREADS_BIND="0-79|80-159|160-239|240-319"

export OMP_NUM_THREADS=$OMP_THREADS_PER_RANK
export MKL_NUM_THREADS=$OMP_THREADS_PER_RANK
export NUMEXPR_MAX_THREADS=$OMP_THREADS_PER_RANK
export OPENBLAS_NUM_THREADS=$OMP_THREADS_PER_RANK
export VECLIB_MAXIMUM_THREADS=$OMP_THREADS_PER_RANK
export GOTO_NUM_THREADS=$OMP_THREADS_PER_RANK

export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=36000

/home/zhangxu/codex/vllm-aarch64-v0.28.0/.venv/bin/python -m vllm.entrypoints.openai.api_server \
    --model "/mnt/models/DeepSeek-V4-Flash-BF16/" \
    --host "127.0.0.1" \
    --port "8004" \
    --gpu-memory-utilization "0.95" \
    --max-model-len "4096" \
    --tensor-parallel-size "$TP_SIZE" \
    --dtype "bfloat16" \
    --trust-remote-code
