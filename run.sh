export LD_PRELOAD=/usr/lib64/libjemalloc.so
export MALLOC_CONF="thp:always,oversize_threshold:2097152,background_thread:true"
export GLOO_DEVICE_TRANSPORT="TCP"
export GLOO_SOCKET_IFNAME="eno1"
export VLLM_CPU_KVCACHE_SPACE=10

export VLLM_CPU_AWQ_USE_FUSED_CPP=0
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=36000

/home/zhangxu/codex/vllm-aarch64-v0.22.0-dsv4/.venv/bin/python -m vllm.entrypoints.openai.api_server \
    --model "/mnt/models/DeepSeek-V4-Flash-BF16/" \
    --host "127.0.0.1" \
    --port "8004" \
    --gpu-memory-utilization "0.95" \
    --max-model-len "4096" \
    --tensor-parallel-size "4" \
    --dtype "bfloat16" \
    --trust-remote-code \
    --enforce-eager