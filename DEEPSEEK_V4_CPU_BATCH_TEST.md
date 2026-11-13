<!-- Copyright (c) 2026 Huawei Technologies Co., Ltd. -->

# DeepSeek V4 CPU 单请求、Batch 与持续压力测试

这套脚本在 `Arm-codex-internal` 的现有环境中，通过 vLLM Completions HTTP
接口比较 BF16、W8A8 INT8 的单请求和 batch 答案，再连续发送 batch32。
每道题只生成一个 A/B/C/D token，测量 prefill、首 token 采样及 HTTP 返回耗时。

使用 MMLU `high_school_geography` 的 5 个 dev 示例，从 test 前 40 题中选择
按部署 tokenizer 编码后最短的 32 题。不是完整 MMLU 评测；完整评测流程见
[DEEPSEEK_V4_CPU_MMLU.md](DEEPSEEK_V4_CPU_MMLU.md)。

## 1. 脚本及数据流

| 脚本 | 用途 |
| --- | --- |
| `scripts/mmlu/prefill_batch/prepare.py` | 生成非 thinking 的 5-shot prompt，核验 BF16/INT8 token ID 一致，保存 32 题和选项 token ID |
| `scripts/mmlu/prefill_batch/compare.py` | 2 次单题预热、32 个串行请求、一次 native batch32、差异题重复请求及重复 batch |
| `scripts/mmlu/prefill_batch/stress.py` | 读取同模型的单请求/batch 基线，持续发送 native batch32，保存逐波原始响应和答案变化 |

每个 native batch32 请求的 `prompt` 是 **32 组独立的 token ID 数组**。
vLLM 调度器决定一次模型执行接收哪些请求，并保留各题的独立序列边界。

- `max_model_len` 限制单条请求的上下文长度。
- `max_num_batched_tokens` 限制一次调度的总 token 数。
- 两个值同时设为 2048、4096、8192，分别建立单请求与 batch 对照。
- 完整序列长度和入队时间会影响合并，模型 batch 不一定恰好填满预算。
- `max_tokens=1`，第一 token 由 prefill 产生，随后请求结束，无后续自回归 decode。

## 2. 现有环境与准备数据

以下命令在测试机的 Bash 中执行。服务环境已安装 vLLM、tokenizers、fused_cpp，
模型、LLVM OpenMP、jemalloc、native CPU allocator、rank0 数值校准表及 BF16
prepack cache 已部署。脚本不安装依赖，也不下载模型。

```bash
VLLM_ROOT="$HOME/final_test/vllm-aarch64-opt"
PY="$HOME/final_test/test/bin/python"
SCRIPTS="$VLLM_ROOT/scripts/mmlu/prefill_batch"
RESULTS="$HOME/final_test/results/prefill-batch-$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$RESULTS/data"
```

从 Hugging Face Dataset Viewer 获取本实验使用的公开数据。保留原始 JSON，
再次测试时复用同一份数据，避免数据顺序变化影响选题：

```bash
curl -fL --retry 2 \
  'https://datasets-server.huggingface.co/rows?dataset=cais%2Fmmlu&config=high_school_geography&split=test&offset=0&length=40' \
  -o "$RESULTS/data/mmlu_geography_test40.json"
curl -fL --retry 2 \
  'https://datasets-server.huggingface.co/rows?dataset=cais%2Fmmlu&config=high_school_geography&split=dev&offset=0&length=5' \
  -o "$RESULTS/data/mmlu_geography_dev5.json"

"$PY" "$SCRIPTS/prepare.py" \
  --data-dir "$RESULTS/data" --output "$RESULTS/cases" \
  --bf16-tokenizer /mnt/models/DeepSeek-V4-Flash-BF16/tokenizer.json \
  --int8-tokenizer /mnt/models/DeepSeek-V4-Flash-INT8/tokenizer.json
```

实测输出：32 题，每题 354–383 tokens，总计 11,743 tokens，A/B/C/D 的
`allowed_token_ids` 为 `[35, 36, 37, 38]`。脚本核验单题加一个答案 token 不超过
2K、总输入超过 8K，以及两种模型 tokenizer 编码一致。输出目录必须是新目录。

数据来源：[cais/mmlu / high_school_geography](https://huggingface.co/datasets/cais/mmlu/tree/main/high_school_geography)。

## 3. 启动服务与机器占用标志

复现本次环境使用以下路径：

```bash
FUSED_ROOT="$HOME/codex/bf16-silu-delivery-fix-20261009/candidate"
RUNTIME="$HOME/final_test/cpu_runtime"
OBSERVER="$HOME/final_test/tp4_acceptance_20261010/bounded-memory-v1/timeout-diagnosis/prefill-path-observer-overlay"
MARKER="$HOME/arm-codex-experiment-status.json"
```

`FUSED_ROOT` 是 SiLU 修复 candidate，对应修复提交 `744855b`。
`RUNTIME/profiles/rank{tp_rank}.json` 的各 rank 使用 rank0 数值。
当前默认使用混合分配器：PyTorch 保留内置 mimalloc，普通分配走 jemalloc。
不再加载 `cpu_allocator_bootstrap` 或独立 allocator 库。以下启动示例已更新为该配置；
文中历史测试结果仍来自当时的自定义 allocator 部署。
`OBSERVER` 是本次实验部署的 attention 路径观测副本，记录 C4 post 的请求数、
token 数、decode token 数和 fused 路径状态；它依赖该机器现有部署。

本次实测基于 vLLM `b7c95b73d4` 加已部署的 allocator/attention 工作区修改。
启动时保存源码路径、commit 和工作区状态，便于区分测试脚本与实际服务版本。

先定义占用标志更新函数，只维护三个字段，并原子替换文件：

```bash
mark_machine() {
  "$PY" - "$MARKER" "$1" "$2" "$RESULTS" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
temporary = path.with_suffix('.tmp')
temporary.write_text(json.dumps({
    'status': sys.argv[2], 'phase': sys.argv[3], 'results_dir': sys.argv[4],
}, ensure_ascii=False, indent=2) + '\n')
temporary.replace(path)
PY
}
```

每种权重、每个预算使用新的服务目录；两种模型共用端口 8018，按顺序运行。
下面示例启动 BF16 2K；随后依次改为 BF16 4K、8K，W8A8 2K、4K、8K。
账号需要已具备 NI=-20 权限：

```bash
variant=bf16                    # INT8 使用 w8a8
budget=2048                     # 依次使用 2048、4096、8192
run_dir="$RESULTS/serve-$variant-$budget"
mkdir "$run_dir"
mark_machine occupied "Starting $variant $budget"

unset OMP_WAIT_POLICY KMP_BLOCKTIME
unset OMP_NUM_THREADS MKL_NUM_THREADS NUMEXPR_MAX_THREADS OPENBLAS_NUM_THREADS
unset VECLIB_MAXIMUM_THREADS GOTO_NUM_THREADS KMP_AFFINITY KMP_TPAUSE
unset DIAG_CONFIG DIAG_CONTROL GLOO_DEVICE_TRANSPORT
unset VLLM_CPU_MOE_PREPACK_EXPORT_DIR VLLM_TORCH_PROFILER_SKIP_TRACE
if [[ "$variant" == bf16 ]]; then
  model=/mnt/models/DeepSeek-V4-Flash-BF16
  export VLLM_CPU_MOE_PREPACKED_DIR=/mnt/models/.cache/zhangxu/DeepSeek-V4-Flash-BF16-moe-prepacked-v028-482e03e
else
  model=/mnt/models/DeepSeek-V4-Flash-INT8
  unset VLLM_CPU_MOE_PREPACKED_DIR
fi
export FUSED_CPP_MOE_PLANNER_PROFILE="$RUNTIME/profiles/rank{tp_rank}.json"
export FUSED_CPP_MOE_HIERARCHICAL_N_SPLIT=0
export VLLM_CPU_MOE_PREPACK_PREFAULT=1
export PYTHONPATH="$FUSED_ROOT/src:$FUSED_ROOT:$OBSERVER:$VLLM_ROOT"
export LD_LIBRARY_PATH=/usr/local/lib
export LD_PRELOAD=/usr/local/lib/libomp.so:/usr/lib64/libjemalloc.so
export MIMALLOC_PURGE_DELAY=-1
export MALLOC_CONF=background_thread:true,metadata_thp:auto,oversize_threshold:0,dirty_decay_ms:-1,muzzy_decay_ms:-1
export GLOO_SOCKET_IFNAME=eno1
export VLLM_TARGET_DEVICE=cpu VLLM_CPU_FUSED_CPP_STRICT=1
export VLLM_CPU_KVCACHE_SPACE=32 VLLM_CPU_NUM_OF_RESERVED_CPU=0
export VLLM_CPU_OMP_THREADS_BIND='0-79|80-159|160-239|240-319'
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=36000
export VLLM_CUSTOM_SCOPES_FOR_PROFILING=1
export PREFILL_PATH_LOG_DIR="$run_dir"

nice -n -20 "$PY" -c 'import os, torch; assert os.getpriority(os.PRIO_PROCESS, 0) == -20'
nice -n -20 "$PY" -c 'import vllm.models.deepseek_v4.cpu.attention as a; print(a.__file__); assert "prefill-path-observer-overlay" in a.__file__' \
  > "$run_dir/source-path.txt"
env | sort | grep -E '^(PYTHONPATH|LD_PRELOAD|LD_LIBRARY_PATH|MIMALLOC_|MALLOC_CONF|GLOO_|VLLM_|FUSED_CPP_)' \
  > "$run_dir/environment.txt"
git -C "$VLLM_ROOT" rev-parse HEAD > "$run_dir/vllm.commit"
git -C "$VLLM_ROOT" status --short > "$run_dir/vllm.status"

nohup nice -n -20 "$PY" -m vllm.entrypoints.openai.api_server \
  --model "$model" --host 127.0.0.1 --port 8018 \
  --served-model-name "dsv4-flash-$variant" \
  --max-model-len "$budget" --max-num-batched-tokens "$budget" \
  --max-num-seqs 128 --tensor-parallel-size 4 --dtype bfloat16 \
  --trust-remote-code --block-size 256 --kv-cache-memory-bytes 34359738368 \
  --gpu-memory-utilization 0.4 --enforce-eager \
  --no-enable-chunked-prefill --no-enable-prefix-caching \
  --numa-bind --numa-bind-node-groups 0 1 2 3 \
  --numa-bind-cpus 0-79 80-159 160-239 240-319 \
  > "$run_dir/server.log" 2>&1 < /dev/null &
echo $! > "$run_dir/server.pid"
tail -f "$run_dir/server.log"
```

看到 `Application startup complete` 后退出 `tail`，服务继续运行。检查健康状态：

```bash
curl -f http://127.0.0.1:8018/health
```

BF16 使用磁盘 prepack cache；INT8 在启动时打包权重，通常需要数分钟。
启动时间不计入请求延迟。

## 4. 单请求与 Batch32 对照

保持该档服务运行，使用第二节准备的相同数据：

```bash
mark_machine occupied "Comparing $variant $budget"
"$PY" "$SCRIPTS/compare.py" \
  --model "dsv4-flash-$variant" --base-url http://127.0.0.1:8018 \
  --cases-dir "$RESULTS/cases" \
  --output "$RESULTS/$variant-$budget" \
  > "$RESULTS/$variant-$budget-client.log" 2>&1
cat "$RESULTS/$variant-$budget/summary.json"
```

脚本固定 temperature=0、seed=0、max_tokens=1、logprobs=4，选项 token 由
`dataset-metadata.json` 读取。校验 usage、每题恰好一个答案 token、有限 logprob、
结束原因和 A/B/C/D 输出，分别统计标准答案得分与相对单请求的一致率。

`summary.json` 的 `batch_consistent` 才表示答案完全一致；程序正常退出仅表示
协议测试完成。标准答案得分不是必须满分的传输校验，也不能据此隐藏答案差异。

完成一档后按第六节停止服务，再启动下一档。8K 对照完成后，可以保持服务运行，
接着执行压力测试。

## 5. 连续压力测试

压力测试固定使用 8K 服务和同模型 8K 对照结果作为基线。每波响应返回后立即发送
下一波，不设置请求间隔；第一波预热排除在正式计时之外。

```bash
mark_machine occupied "Pressure $variant 8192"
"$PY" "$SCRIPTS/stress.py" \
  --model "dsv4-flash-$variant" --base-url http://127.0.0.1:8018 \
  --cases-dir "$RESULTS/cases" --baseline "$RESULTS/$variant-8192" \
  --seconds 600 --output "$RESULTS/pressure-$variant-8192" \
  > "$RESULTS/pressure-$variant-8192-client.log" 2>&1
cat "$RESULTS/pressure-$variant-8192/summary.json"
```

开始最后一波时可能尚未满 600 秒，脚本等待该波完整返回，因此正式时长可能略长。
请求超时为 600 秒；HTTP/格式校验失败会保存原始响应及错误，停止本阶段并返回非零。

判定时同时查看：

- `duration_reached=true`、`failed_waves=0`：持续压力阶段完整运行，响应校验正常。
- `same_as_single=completed_questions`：全部与同模型单请求基线一致。
- `same_as_previous_batch=completed_questions`：全部与此前同模型 batch 基线一致。
- `gold_correct`：标准答案正确数；同一组 32 题循环，题目请求数不是不同题目数。

一致率或标准答案得分不足时，仍会保存结果，不能把退出码 0 写成“全部答案正确”。

## 6. 停止服务与释放标志

每档测试结束后停止该服务，核验记录的 API、EngineCore 和 rank PID 都已退出。
只对该服务日志记录的 PID 操作：

```bash
"$PY" - "$run_dir" <<'PY'
import os
import re
import signal
import sys
import time
from pathlib import Path

run = Path(sys.argv[1])
log = (run / 'server.log').read_text()
pids = {int((run / 'server.pid').read_text())}
pids.update(map(int, re.findall(r'Worker_TP\d pid=(\d+)', log)))
pids.update(map(int, re.findall(r'EngineCore(?:_DP\d+)? pid=(\d+)', log)))
for pid in sorted(pids):
    path = Path(f'/proc/{pid}')
    if path.exists():
        assert path.stat().st_uid == os.getuid()
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
end = time.monotonic() + 20
while time.monotonic() < end and any(Path(f'/proc/{p}').exists() for p in pids):
    time.sleep(0.5)
for pid in sorted(pids):
    if Path(f'/proc/{pid}').exists():
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
end = time.monotonic() + 5
while time.monotonic() < end and any(Path(f'/proc/{p}').exists() for p in pids):
    time.sleep(0.1)
alive = [p for p in sorted(pids) if Path(f'/proc/{p}').exists()]
assert not alive, f'Recorded PIDs still present: {alive}'
print('Stopped recorded PIDs:', sorted(pids))
PY
ss -ltn sport = :8018
```

确认端口没有监听、记录的 PID 均已退出后，再启动下一档。全部实验结束或主动
中断并完成清理后，更新标志：

```bash
mark_machine released 'Tests stopped; results saved'
cat "$MARKER"
```

## 7. 结果文件与实际 Batch 核验

| 文件 | 内容 |
| --- | --- |
| `cases/cases32.jsonl`、`dataset-metadata.json` | prompt、token IDs、标准答案、选项 IDs 和长度 |
| `<模型>-<预算>/protocol.json` | 对照参数 |
| 同目录 `single.json`、`batch32.json` | 原始 Completions 响应及逐题结果 |
| 同目录 `repeat-singles.json`、`repeat-batch32.json` | 差异题和 batch 重复结果 |
| 同目录 `summary.json` | 标准答案得分、一致率和耗时 |
| `pressure-*/wave-*.json` | 每波原始响应、错误、两类基线比较及时间范围 |
| `pressure-*/summary.json` | 正式时长、波次、延迟范围、失败数及累计一致率 |
| `serve-*/c4-post-*.jsonl` | 每 rank、每 C4 层的实际请求数、token 数及 fused/decode 状态 |

路径日志每个模型步骤会记录多个 C4 层，且有四个 rank；统计真实调度次数时选一个
rank、固定一个 `prefix`，按请求的 `start_unix`/`end_unix` 过滤，避免重复计数。
四个 rank 的该层请求数/token 数序列应相同。每波所有步骤的请求数合计为 32，
输入 token 数合计为 11,743；每步不超过调度预算。纯 prefill 应为
`decode_tokens=0`、`c4_fused_post_used=true`。

## 8. 本次实测结果

TP4 × 80 核/rank、KV32GiB/rank、LLVM OpenMP、jemalloc、NI=-20，
SiLU 修复 candidate 和 rank0 数值校准表；数据及参数见上述步骤。

| 模型 | 预算 | 单请求答对 | Batch答对 | 与单请求一致 | 串行32题/s | Batch32/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BF16 | 2K | 28/32 | 29/32 | 31/32 | 70.57 | 36.87 |
| BF16 | 4K | 28/32 | 29/32 | 31/32 | 72.97 | 34.91 |
| BF16 | 8K | 28/32 | 29/32 | 31/32 | 68.37 | 33.56 |
| INT8 | 2K | 27/32 | 27/32 | 32/32 | 59.82 | 24.06 |
| INT8 | 4K | 27/32 | 27/32 | 32/32 | 59.47 | 20.44 |
| INT8 | 8K | 27/32 | 27/32 | 32/32 | 62.14 | 21.90 |

主要模型 batch 分别达到 5 题/~1.8K、11 题/~4K、22 题/~8K tokens。
各 rank 边界相同，观测 C4 post 全部 fused，decode tokens 为 0。

| 模型 | 正式时长/s | 波次 | 题目请求数 | 失败波次 | 平均Batch32/s | 最慢/s | 与单请求一致 | 与此前Batch一致 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BF16 | 605.24 | 18 | 576 | 0 | 33.62 | 36.16 | 558/576 | 576/576 |
| INT8 | 608.01 | 29 | 928 | 0 | 20.96 | 24.56 | 928/928 | 928/928 |

BF16 三档均有 `test014` 从单请求 A 变为 batch B，标准答案 B；重复及压力请求
保持这个差异，具体算子原因尚未定位。INT8 三档及持续压力与单请求完全一致。
MMLU 标准答案得分并非满分，不能写成“无回答错误”的验收通过。

原始实验目录为
`$HOME/final_test/tp4_acceptance_20261010/mmlu-prefill-batch-v1/`，保留了原始
请求、四 rank 路径、汇总、SiLU 版本和进程清理证据。本次脚本整理后的路径参数化
使用同样的 payload、prompt 和评分规则；这些实测数字来自此前保存的实验结果。
