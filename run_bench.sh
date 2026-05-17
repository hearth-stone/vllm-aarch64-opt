#!/bin/bash

# 告诉系统，这个脚本遇到严重错误也不要随便退出
#输入nohup bash run_bench.sh &
set +e 

LOG_FILE="nightly_bench_70B.log"

echo "=================================================" > $LOG_FILE
echo "🚀 70B 巨兽通宵无人值守压测启动: $(date)" | tee -a $LOG_FILE
echo "=================================================" | tee -a $LOG_FILE

# 1. 封装一个“核弹清理”函数
cleanup() {
    echo "[$(date)] 🧹 开始自动清理上一轮的战场..." | tee -a $LOG_FILE
    pkill -9 -u $(whoami) -f vllm >/dev/null 2>&1
    pkill -9 -u $(whoami) python >/dev/null 2>&1
    rm -rf /dev/shm/*$(whoami)* >/dev/null 2>&1
    sleep 5 # 等待 5 秒，让操作系统彻底回收物理内存
    echo "[$(date)] ✨ 战场清理完毕！" | tee -a $LOG_FILE
}

# 2. 注入所有的保命和极限调优环境变量
export OMP_NUM_THREADS=8
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_RPC_TIMEOUT=120
export VLLM_SHARED_MEMORY_BROADCAST_TIMEOUT=120

# 3. 自动遍历你想测的配置 (比如这里我们让它通宵测 512, 1024, 2048 三个长度)
for seq_len in 512 1024 2048; do
    
    # 每次跑之前，先无条件清场，绝不让上一次的僵尸影响这一次
    cleanup

    echo -e "\n▶️ [$(date)] 正在测试 Input Length: ${seq_len} (包含 10 分钟超长热身) ..." | tee -a $LOG_FILE

    # 4. 执行核心命令，并把所有的输出（包含报错）追加保存到日志文件里
    numactl --localalloc vllm bench latency \
        --model /mnt/models/DeepSeek-R1-Distill-Llama-70B \
        --tensor-parallel-size 4 \
        --batch-size 1 \
        --input-len ${seq_len} \
        --output-len 1 \
        --gpu-memory-utilization 0.6 \
        --enforce-eager >> $LOG_FILE 2>&1
        
    # $? 是上一个命令的退出码，0代表成功
    if [ $? -eq 0 ]; then
        echo "✅ [$(date)] 长度 ${seq_len} 测试完美结束！" | tee -a $LOG_FILE
    else
        echo "❌ [$(date)] 长度 ${seq_len} 测试崩溃了，详情请看上面的日志。" | tee -a $LOG_FILE
    fi

done

# 跑完最后一遍，干干净净地把内存还给系统
cleanup
echo "🎉 全部测试跑完了！输入cat nightly_bench_70B.log 看数据了！" | tee -a $LOG_FILE