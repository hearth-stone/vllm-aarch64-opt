# DeepSeek V4 CPU 实验运行说明

本文记录 `dsv4-arm-cpu-v0.28.0` 在 `Arm-codex-internal` 上的可复现实验方法。
命令默认使用 TP4、每个 rank 80 个 CPU、严格 NUMA 绑定、fused_cpp strict
模式和已经生成的 MoE prepack cache。

## 1. 路径与版本

本地工作树：

```text
/Users/zhangxu/Codes/vllm-aarch64/vllm-aarch64-v0.28.0
/Users/zhangxu/Codes/vllm-aarch64/fused_cpp-482e03e-v028
```

远程运行目录：

```text
/home/zhangxu/codex/vllm-aarch64-v0.28.0-dsv4
/home/zhangxu/codex/fused_cpp-482e03e-v028
```

分支和关键提交：

```text
vLLM branch:      dsv4-arm-cpu-v0.28.0
vLLM mHC:         a81aa5c4a0 Integrate fused DeepSeek V4 CPU mHC
vLLM NUMA:        c184b45afd Preserve strict NUMA worker binding

fused_cpp branch: dsv4-arm-cpu-v0.28.0
fused_cpp mHC:    377486a Add fused DeepSeek V4 mHC kernels
fused_cpp W8A8:   109d87c Fix W8A8 MoE workspace reuse
```

远程 vLLM 目录是部署副本，不依赖其 Git 元数据。实验实际加载：

```text
/home/zhangxu/codex/vllm-aarch64-v0.28.0-dsv4/vllm/_C.abi3.so
/home/zhangxu/codex/fused_cpp-482e03e-v028/src/fused_cpp/_C.cpython-312-aarch64-linux-gnu.so
/home/zhangxu/codex/fused_cpp-482e03e-v028/src/fused_cpp/_moe_C.cpython-312-aarch64-linux-gnu.so
```

## 2. 模型、测试集与 cache

```text
BF16 model: /mnt/models/DeepSeek-V4-Flash-BF16
INT8 model: /mnt/models/DeepSeek-V4-Flash-INT8

BF16 MoE cache:
/mnt/models/.cache/zhangxu/DeepSeek-V4-Flash-BF16-moe-prepacked-v028-482e03e

INT8 MoE cache:
/mnt/models/.cache/zhangxu/DeepSeek-V4-Flash-INT8-moe-prepacked-v028-482e03e

2048-token cases:
/mnt/models/.cache/zhangxu/benchmarks/fused-attn80-continuous30-20260821/cases31.jsonl

rank-local planner profile:
/mnt/models/.cache/zhangxu/fused_cpp-profiles/arm-codex-tp4-runtime-rank{local_rank}.json
```

标准 runner 是：

```text
tests/v1/e2e/generation_sampling/generation_prefill_suite.py
```

如果远程 v0.28 部署副本没有同步 `tests/`，可使用现有 runner 文件：

```text
/home/zhangxu/codex/vllm-aarch64-v0.22.0-dsv4/tests/v1/e2e/generation_sampling/generation_prefill_suite.py
```

该文件只负责构造 `LLM`、发送固定 token IDs、计时和写 JSON；`PYTHONPATH`、
Python 环境、vLLM 源码和二进制仍全部指向 v0.28。

## 3. 实验前检查

不要在模型下载、编译或其他大内存任务运行时测性能：

```bash
ssh Arm-codex-internal '
  uptime
  pgrep -af "hf download|vllm|ninja|cmake" || true
  ps -eo pid,psr,pcpu,pmem,stat,comm,args --sort=-pcpu | head -20
  sensors
  for cpu in 0 80 160 240; do
    printf "cpu%s " "$cpu"
    cat /sys/devices/system/cpu/cpu${cpu}/cpufreq/scaling_cur_freq
  done
'
```

已观测到 `hf download DeepSeek-V4-Pro-Base` 使用 67 个线程、跨 0-319 CPU
调度并突发写盘时，BF16 从无干扰的约 4.86 秒下降到约 5.55 秒。因此正式结果前
必须确认后台任务为空。

## 4. 公共运行环境

以下变量用于所有 BF16/INT8 实验：

```bash
VLLM_ROOT=/home/zhangxu/codex/vllm-aarch64-v0.28.0-dsv4
FUSED_ROOT=/home/zhangxu/codex/fused_cpp-482e03e-v028
RUNNER=/home/zhangxu/codex/vllm-aarch64-v0.22.0-dsv4/tests/v1/e2e/generation_sampling/generation_prefill_suite.py
CASES=/mnt/models/.cache/zhangxu/benchmarks/fused-attn80-continuous30-20260821/cases31.jsonl
PLANNER='/mnt/models/.cache/zhangxu/fused_cpp-profiles/arm-codex-tp4-runtime-rank{local_rank}.json'
```

公共环境：

```bash
env \
  VLLM_TARGET_DEVICE=cpu \
  LD_LIBRARY_PATH=/opt/llvm-22/lib/aarch64-unknown-linux-gnu \
  LD_PRELOAD=/opt/llvm-22/lib/aarch64-unknown-linux-gnu/libomp.so:/usr/lib64/libjemalloc.so \
  MALLOC_CONF=thp:always,oversize_threshold:2097152,background_thread:true \
  GLOO_DEVICE_TRANSPORT=TCP \
  GLOO_SOCKET_IFNAME=eno1 \
  VLLM_CPU_KVCACHE_SPACE=5 \
  'VLLM_CPU_OMP_THREADS_BIND=0-79|80-159|160-239|240-319' \
  VLLM_CPU_NUM_OF_RESERVED_CPU=0 \
  VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=36000 \
  VLLM_CPU_FUSED_CPP_STRICT=1 \
  "FUSED_CPP_MOE_PLANNER_PROFILE=$PLANNER" \
  FUSED_CPP_MOE_HIERARCHICAL_N_SPLIT=0 \
  VLLM_CPU_MOE_PREPACK_PREFAULT=1 \
  VLLM_CPU_MOE_PREPACKED_DIR="$CACHE_DIR" \
  PYTHONPATH="$FUSED_ROOT/src:$VLLM_ROOT" \
  "$VLLM_ROOT/.venv/bin/python" "$RUNNER" ...
```

注意：

- 必须显式设置 `VLLM_TARGET_DEVICE=cpu`。远程部署副本没有完整 editable-package
  metadata 时，省略该变量可能得到 `Device string must not be empty`。
- strict 模式下，目标 MoE、Attention W8A8 和 fused_cpp 路径不可用时直接失败，
  不允许静默 fallback。
- `--block-size 256` 是 DeepSeek V4 CPU sparse MLA 的必需值。
- NUMA 绑定为 rank0=`0-79`、rank1=`80-159`、rank2=`160-239`、
  rank3=`240-319`，内存分别绑定 node 0/1/2/3。

## 5. 快速可比 E2E：warmup + 3 条

用于日常回归定位。warmup 不计时，之后每条请求间隔 10 秒。

### 5.1 BF16

```bash
ssh Arm-codex-internal '
set -euo pipefail
VLLM_ROOT=/home/zhangxu/codex/vllm-aarch64-v0.28.0-dsv4
FUSED_ROOT=/home/zhangxu/codex/fused_cpp-482e03e-v028
RUNNER=/home/zhangxu/codex/vllm-aarch64-v0.22.0-dsv4/tests/v1/e2e/generation_sampling/generation_prefill_suite.py
CASES_SRC=/mnt/models/.cache/zhangxu/benchmarks/fused-attn80-continuous30-20260821/cases31.jsonl
OUT=/home/zhangxu/codex/dsv4-v028-results/bf16-quick
MODEL=/mnt/models/DeepSeek-V4-Flash-BF16
CACHE_DIR=/mnt/models/.cache/zhangxu/DeepSeek-V4-Flash-BF16-moe-prepacked-v028-482e03e
PLANNER="/mnt/models/.cache/zhangxu/fused_cpp-profiles/arm-codex-tp4-runtime-rank{local_rank}.json"

mkdir -p "$OUT"
head -n 4 "$CASES_SRC" > "$OUT/cases4.jsonl"
cd "$VLLM_ROOT"
env \
  VLLM_TARGET_DEVICE=cpu \
  LD_LIBRARY_PATH=/opt/llvm-22/lib/aarch64-unknown-linux-gnu \
  LD_PRELOAD=/opt/llvm-22/lib/aarch64-unknown-linux-gnu/libomp.so:/usr/lib64/libjemalloc.so \
  MALLOC_CONF=thp:always,oversize_threshold:2097152,background_thread:true \
  GLOO_DEVICE_TRANSPORT=TCP GLOO_SOCKET_IFNAME=eno1 \
  VLLM_CPU_KVCACHE_SPACE=5 \
  "VLLM_CPU_OMP_THREADS_BIND=0-79|80-159|160-239|240-319" \
  VLLM_CPU_NUM_OF_RESERVED_CPU=0 \
  VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=36000 \
  VLLM_CPU_FUSED_CPP_STRICT=1 \
  "FUSED_CPP_MOE_PLANNER_PROFILE=$PLANNER" \
  FUSED_CPP_MOE_HIERARCHICAL_N_SPLIT=0 \
  VLLM_CPU_MOE_PREPACK_PREFAULT=1 \
  VLLM_CPU_MOE_PREPACKED_DIR="$CACHE_DIR" \
  PYTHONPATH="$FUSED_ROOT/src:$VLLM_ROOT" \
  "$VLLM_ROOT/.venv/bin/python" "$RUNNER" \
    --model "$MODEL" \
    --cases "$OUT/cases4.jsonl" \
    --output "$OUT/result.jsonl" \
    --tensor-parallel-size 4 \
    --max-model-len 2100 \
    --block-size 256 \
    --kv-cache-memory-bytes 5368709120 \
    --numa-bind-cpus "0-79|80-159|160-239|240-319" \
    --numa-bind-nodes 0,1,2,3 \
    --expected-prompt-tokens 2048 \
    --min-measured-cases 3 \
    --warmup-case-index 0 \
    --exclude-warmup-case \
    --request-interval-seconds 10 \
    --max-tokens 1 \
  2>&1 | tee "$OUT/run.log"
'
```

2026-08-27 无后台干扰的结果：

```text
5.0780, 4.8227, 4.6847 秒
平均 4.8618 秒，中位数 4.8227 秒
```

### 5.2 INT8

复用 BF16 命令，仅修改：

```bash
MODEL=/mnt/models/DeepSeek-V4-Flash-INT8
CACHE_DIR=/mnt/models/.cache/zhangxu/DeepSeek-V4-Flash-INT8-moe-prepacked-v028-482e03e
OUT=/home/zhangxu/codex/dsv4-v028-results/int8-quick
```

模型 config 自动选择 INT8/W8A8 路径，不额外传 quantization 参数。历史无 profiler
E2E 约为 2.88 秒；正式结果必须重新按相同机器状态复测。

## 6. 正式性能门禁

BF16 和 INT8 分开执行，每个候选和 baseline 各运行 3 个独立 engine。每个 engine：

1. 运行一条 warmup，排除耗时。
2. warmup 后等待 10 秒。
3. 运行 5 条固定 2048-token、greedy、`max_tokens=1` 请求。
4. 每条请求之间等待 10 秒。
5. 关闭 profiler 和额外诊断。

准备 1 条 warmup + 5 条 measured：

```bash
head -n 6 "$CASES" > "$OUT/cases6.jsonl"
```

将快速命令中的参数改为：

```text
--cases cases6.jsonl
--min-measured-cases 5
--request-interval-seconds 10
```

每个 engine 得到 5 条平均值，最终指标取 3 个 engine 平均值的中位数：

```text
candidate / fresh_current_baseline <= 1.02
```

推荐 A/B 交错执行：

```text
baseline-1, candidate-1, baseline-2, candidate-2,
baseline-3, candidate-3, final-baseline-drift-check
```

每个 engine 使用独立输出目录。不要追加到旧 `result.jsonl`。

## 7. 正确性检查

runner 使用：

```text
temperature=0
max_tokens=1
ignore_eos=True
repetition_penalty=1.1
seed=42
```

逐条比较 `generated_token_ids`，BF16/INT8 候选必须与各自 baseline 一致。当前
BF16 三条参考输出是：

```text
zh2048-002 -> [21773]
zh2048-003 -> [42078]
zh2048-004 -> [16118]
```

算子级正确性另外比较：

```text
max absolute error
relative L2
cosine similarity
```

涉及量化转换时，不应只检查 greedy token。

## 8. Component profile

profile 必须使用独立 engine，结果不参与性能门禁。准备 1 条 warmup + 1 条 measured：

```bash
head -n 2 "$CASES" > "$OUT/cases2.jsonl"
```

在快速命令中增加/修改：

```text
--cases cases2.jsonl
--min-measured-cases 1
--profile-dir "$OUT/profile"
--request-interval-seconds 10
```

runner 在 warmup 完成后调用 `llm.start_profile()`，因此 warmup 不进入 trace。主要
record scope：

```text
vllm::deepseek_v4_moe
vllm::deepseek_v4_attention
vllm::deepseek_v4_attention/input_fused_cpp
vllm::deepseek_v4_attention/post_fused_cpp
vllm::deepseek_v4_attention/sparse_fused_cpp
vllm::deepseek_v4_attention/output_inv_rope_woa_fused_cpp
vllm::deepseek_v4_attention/output_wo_b
vllm::deepseek_v4_attention/output_wo_b_w8a8
```

profile 需要确认：

- 43 层 Flash 模型全部进入目标 Attention/MoE fused scope。
- BF16/W8A8 MoE 共 43 次。
- INT8 模型目标 Attention projection 进入 W8A8。
- prepack cache 命中。
- strict 模式无 fallback。
- 四个 TP rank 都使用各自完整 80 核。

## 9. 频率和温度采样

需要记录机器状态时，在 runner 命令增加：

```text
--frequency-probe-cpus "0-79|80-159|160-239|240-319"
--frequency-probe-interval-seconds 0.05
--thermal-probe-zones "0,1,2,3,4,5,6,7"
```

结果写入每条 JSON 的 `frequency_mhz` 和 `thermal_c`。如果 sysfs 只报告固定
2.9GHz，仍需结合请求时间和 `sensors` 判断机器状态；不能仅凭
`scaling_cur_freq` 排除 thermal throttling。

## 10. 连续压力测试

连续测试用于观察热稳定性，不作为正式性能门禁：

```text
1 条 warmup + 30 条 measured
--request-interval-seconds 0
```

准备 cases：

```bash
head -n 31 "$CASES" > "$OUT/cases31.jsonl"
```

2026-08-25 的 BF16 连续测试从前 10 条平均 5.60 秒恶化到最后 10 条平均
6.97 秒，最后 5 条平均 7.24 秒。这个实验说明必须保留 10 秒间隔协议，并在
性能门禁前清理后台负载。

## 11. 结果文件

runner 生成：

```text
result.jsonl
result.jsonl.summary.json
run.log
profile/                 # 仅使用 --profile-dir 时存在
```

快速查看：

```bash
cat "$OUT/result.jsonl.summary.json"
```

或使用项目虚拟环境重算平均/中位数：

```bash
"$VLLM_ROOT/.venv/bin/python" -c '
import json, statistics, sys
values = [json.loads(line)["duration_s"] for line in open(sys.argv[1]) if line.strip()]
print("count", len(values))
print("mean", statistics.mean(values))
print("median", statistics.median(values))
print("min", min(values))
print("max", max(values))
' "$OUT/result.jsonl"
```

## 12. 常见问题

### `Device string must not be empty`

显式设置：

```text
VLLM_TARGET_DEVICE=cpu
```

### prepack cache 未命中

检查模型、TP size、rank、shape、dtype、quant mode 和 fused_cpp ABI。strict 模式
应直接失败，不要允许读取原 expert tensor 后静默重建。

### 结果突然从约 4.9 秒变为 5.5 秒以上

首先检查：

```bash
pgrep -af "hf download|vllm|ninja|cmake"
ps -eo pid,psr,pcpu,pmem,stat,comm,args --sort=-pcpu | head -20
iostat -dx 1 3
sensors
```

相同代码和 prompt 在后台 67 线程模型下载期间平均约 5.55 秒，下载结束、机器
空闲后恢复到 4.86 秒。

### 远程 v0.28 没有 runner

使用 v0.22 路径下的同一 runner 文件，同时保持 `PYTHONPATH` 和 Python executable
指向 v0.28。不要将 v0.22 包路径加入 `PYTHONPATH`。
