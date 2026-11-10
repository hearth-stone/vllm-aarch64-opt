# DeepSeek V4 Flash CPU MMLU 评测

在 `arm-codex-internal` 上评测 BF16 和 W8A8 INT8，保存 MMLU 汇总准确率和逐题结果。第 1–3 节完成一次安装和数据准备。第 4、5 节分别是独立的 BF16、INT8 评测流程，可以各自在新的 Bash 会话中执行；每一节内的命令在该会话中按顺序执行。两种服务共用端口 8004，运行下一种前先停止当前服务。

## 系统如何工作

两套 Python 环境分工如下：

| 目录（均在 `$HOME/final_test` 下） | 用途 |
| --- | --- |
| `vllm-aarch64-opt/`、`fused_cpp/` | 服务代码和 CPU 内核 |
| `test/` | 安装 vLLM、fused_cpp，加载模型并提供 HTTP 服务 |
| `mmlu/.venv/` | 安装 lm-evaluation-harness，读取题目并请求服务 |
| `mmlu/cais-mmlu/` | 本地 MMLU Parquet 数据 |
| `mmlu/tasks/mmlu-local/` | 指向本地数据的评测任务配置 |
| `results/` | 服务日志、评分接口响应、评测结果 |

数据流：本地题目 → lm-eval 构造 5-shot 提示和候选答案 → vLLM Completions 接口返回选项的 token logprobs → lm-eval 评分 → JSON/JSONL。评测器使用服务端 tokenizer，不在 MMLU 环境里加载模型。

本流程固定使用测试机的以下约定：

- Linux AArch64、SVE 256 位；GCC/G++ ≥ 12.3。
- TP4，每个 worker 使用 NUMA 4/5/6/7 中的一组 40 核：`160-199`、`200-239`、`240-279`、`280-319`。
- 模型已在 `/mnt/models/DeepSeek-V4-Flash-BF16/` 和 `/mnt/models/DeepSeek-V4-Flash-INT8/`。
- 服务地址 `http://127.0.0.1:8004`，上下文长度 4096；BF16 与 INT8 依次运行。
- 评测固定为 5-shot、单请求并发；评测器和数据集固定到下面的提交。

## 1. 安装服务环境

测试机已有系统依赖。重装系统时用这一组命令准备：

```bash
sudo dnf install -y gcc gcc-c++ jemalloc numactl numactl-devel \
  cmake ninja-build git git-lfs curl
gcc --version
g++ --version
```

克隆两个交付分支：

```bash
mkdir -p "$HOME/final_test"
cd "$HOME/final_test"
git clone -b dsv4-arm-cpu-v0.28.0 --single-branch \
  https://github.com/hearth-stone/vllm-aarch64-opt.git
git clone -b feat/fused_cpp/dsv4_v028_aot_delivery --single-branch \
  https://github.com/hearth-stone/fused_cpp.git

export VLLM_ROOT="$HOME/final_test/vllm-aarch64-opt"
export FUSED_CPP_SRC="$HOME/final_test/fused_cpp/src"
```

安装 uv 并创建服务环境；已有 uv 时跳过前两行：

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"
uv python install 3.12
uv venv --python 3.12 "$HOME/final_test/test"
source "$HOME/final_test/test/bin/activate"
export VLLM_PYTHON="$HOME/final_test/test/bin/python"

cd "$VLLM_ROOT"
uv pip install -r requirements/build/cpu.txt --torch-backend cpu \
  --index-strategy unsafe-best-match \
  --default-index https://mirrors.aliyun.com/pypi/simple/
uv pip install -r requirements/cpu.txt --torch-backend cpu \
  --index-strategy unsafe-best-match \
  --default-index https://mirrors.aliyun.com/pypi/simple/
VLLM_TARGET_DEVICE=cpu MAX_JOBS=16 uv pip install -e . --no-build-isolation
uv pip uninstall torchcodec

cd "$HOME/final_test/fused_cpp"
FUSED_CPP_SVE_VECTOR_BITS=256 MAX_JOBS=16 \
  uv pip install -e . --no-build-isolation --no-deps
uv pip install pytest tblib
```

移除 `torchcodec` 是因为测试机装入的 wheel 导入时缺少 `libnvrtc.so.13`；这套流程只评测文本。

启动脚本要求传入 `VLLM_ROOT`、`VLLM_PYTHON`、`FUSED_CPP_SRC`。这三个路径在已有的 `codex` 部署和新建的 `final_test` 环境中不同，因此由调用者指定。NUMA、网卡 `eno1`、jemalloc 路径和模型位置已经固定在启动脚本里，无需另行配置。

## 2. 验证代码安装

```bash
cd "$HOME/final_test/fused_cpp"
PYTHONPATH="$FUSED_CPP_SRC:$VLLM_ROOT" "$VLLM_PYTHON" -m pytest -q -rs tests/

mkdir -p "$VLLM_ROOT/model_configs/DeepSeek-V4-Flash-BF16"
cp /mnt/models/DeepSeek-V4-Flash-BF16/config.json \
  "$VLLM_ROOT/model_configs/DeepSeek-V4-Flash-BF16/config.json"
cd "$VLLM_ROOT"
VLLM_TARGET_DEVICE=cpu PYTHONPATH="$FUSED_CPP_SRC:$VLLM_ROOT" \
  "$VLLM_PYTHON" -m pytest -q -rs \
  tests/models/test_deepseek_v4_cpu_v028.py \
  tests/model_executor/test_cpu_unquantized_gemm_dispatch.py
```

当前交付版实测：fused_cpp 为 `1582 passed, 1 skipped`，vLLM 为 `45 passed`。唯一的 skip 测试“native backend 不可用”时的报错；本机构建已有 native backend，所以跳过。

## 3. 安装 MMLU 环境与数据

MMLU 环境单独创建，不把 lm-eval 装进服务环境。`MMLU_PYTHON` 是创建环境后用于执行命令的 shell 变量，没有预设值。

```bash
mkdir -p "$HOME/final_test/mmlu"
cd "$HOME/final_test/mmlu"
git clone https://github.com/EleutherAI/lm-evaluation-harness.git
git -C lm-evaluation-harness checkout c1c4bea3777f73e188395264083adcf454913344
uv venv --python 3.12 .venv
MMLU_PYTHON="$HOME/final_test/mmlu/.venv/bin/python"
uv pip install --python "$MMLU_PYTHON" -e './lm-evaluation-harness[api]'

GIT_LFS_SKIP_SMUDGE=1 git clone \
  https://hf-mirror.com/datasets/cais/mmlu cais-mmlu
GIT_LFS_SKIP_SMUDGE=1 git -C cais-mmlu checkout \
  c30699e8356da336a370243923dbaf21066bb9fe
"$MMLU_PYTHON" "$VLLM_ROOT/scripts/mmlu/download_dataset.py" \
  --dataset-dir "$HOME/final_test/mmlu/cais-mmlu" \
  --hf-cli "$HOME/final_test/mmlu/.venv/bin/hf"
"$MMLU_PYTHON" "$VLLM_ROOT/scripts/mmlu/check_dataset.py" \
  --dataset-dir "$HOME/final_test/mmlu/cais-mmlu"
"$MMLU_PYTHON" "$VLLM_ROOT/scripts/mmlu/prepare_tasks.py" \
  --harness-dir "$HOME/final_test/mmlu/lm-evaluation-harness" \
  --dataset-dir "$HOME/final_test/mmlu/cais-mmlu" \
  --task-dir "$HOME/final_test/mmlu/tasks/mmlu-local"
```

测试机连接 `huggingface.co` 曾超时，所以这里直接使用已验证的镜像。检查应报告 176 个 Parquet 文件，`abstract_algebra` 的 `dev=5`、`test=100`。任务准备脚本将 lm-eval 的 MMLU 配置改为本地数据路径，评测时离线读取。

## 4. BF16 + MMLU

前提是第 1–3 节已完成，端口 8004 空闲。下面重新设置本节所需的环境变量，不依赖安装时或另一种模型评测时的 shell 状态；无需激活虚拟环境。启动脚本自动完成 MoE 校准，不需要手动准备 profile。

```bash
export VLLM_ROOT="$HOME/final_test/vllm-aarch64-opt"
export VLLM_PYTHON="$HOME/final_test/test/bin/python"
export FUSED_CPP_SRC="$HOME/final_test/fused_cpp/src"
MMLU_PYTHON="$HOME/final_test/mmlu/.venv/bin/python"
model_id=dsv4-flash-bf16
run_dir="$HOME/final_test/results/bf16-$(date -u +%Y%m%dT%H%M%SZ)"

mkdir -p "$run_dir"
nohup bash "$VLLM_ROOT/run_numa_4567_bf16_autocalib.sh" \
  --served-model-name "$model_id" \
  --enable-tokenizer-info-endpoint --no-enable-prefix-caching \
  > "$run_dir/server.log" 2>&1 < /dev/null &
echo $! > "$run_dir/server.pid"
tail -f "$run_dir/server.log"
```

模型加载和权重准备需要数分钟。看到 `Application startup complete` 后按 Ctrl+C 退出日志查看，服务继续运行。检查评分接口：

```bash
"$MMLU_PYTHON" "$VLLM_ROOT/scripts/mmlu/check_api.py" \
  --model "$model_id" --output-dir "$run_dir"
```

成功时打印 `Serving: dsv4-flash-bf16; prompt logprobs: 33 tokens`。然后运行 `abstract_algebra` 的两题 smoke：

```bash
"$MMLU_PYTHON" "$VLLM_ROOT/scripts/mmlu/run_eval.py" \
  --lm-eval "$HOME/final_test/mmlu/.venv/bin/lm-eval" \
  --harness-dir "$HOME/final_test/mmlu/lm-evaluation-harness" \
  --dataset-dir "$HOME/final_test/mmlu/cais-mmlu" \
  --task-dir "$HOME/final_test/mmlu/tasks/mmlu-local" \
  --model "$model_id" --output-dir "$run_dir" --mode smoke
```

smoke 通过后运行完整 MMLU：

```bash
"$MMLU_PYTHON" "$VLLM_ROOT/scripts/mmlu/run_eval.py" \
  --lm-eval "$HOME/final_test/mmlu/.venv/bin/lm-eval" \
  --harness-dir "$HOME/final_test/mmlu/lm-evaluation-harness" \
  --dataset-dir "$HOME/final_test/mmlu/cais-mmlu" \
  --task-dir "$HOME/final_test/mmlu/tasks/mmlu-local" \
  --model "$model_id" --output-dir "$run_dir" --mode full
```

结果保存在本节的 `run_dir`。评测完成后停止服务；只运行 smoke 时也用这组命令停止：

```bash
kill "$(cat "$run_dir/server.pid")"
while kill -0 "$(cat "$run_dir/server.pid")" 2>/dev/null; do sleep 1; done
```

## 5. W8A8 INT8 + MMLU

前提是第 1–3 节已完成，端口 8004 空闲。下面重新设置本节所需的环境变量，不依赖安装时或另一种模型评测时的 shell 状态；无需激活虚拟环境。启动脚本自动完成 MoE 校准，不需要手动准备 profile。

```bash
export VLLM_ROOT="$HOME/final_test/vllm-aarch64-opt"
export VLLM_PYTHON="$HOME/final_test/test/bin/python"
export FUSED_CPP_SRC="$HOME/final_test/fused_cpp/src"
MMLU_PYTHON="$HOME/final_test/mmlu/.venv/bin/python"
model_id=dsv4-flash-int8
run_dir="$HOME/final_test/results/int8-$(date -u +%Y%m%dT%H%M%SZ)"

mkdir -p "$run_dir"
nohup bash "$VLLM_ROOT/run_numa_4567_w8a8_autocalib.sh" \
  --served-model-name "$model_id" \
  --enable-tokenizer-info-endpoint --no-enable-prefix-caching \
  > "$run_dir/server.log" 2>&1 < /dev/null &
echo $! > "$run_dir/server.pid"
tail -f "$run_dir/server.log"
```

模型加载和权重准备需要数分钟。看到 `Application startup complete` 后按 Ctrl+C 退出日志查看，服务继续运行。检查评分接口：

```bash
"$MMLU_PYTHON" "$VLLM_ROOT/scripts/mmlu/check_api.py" \
  --model "$model_id" --output-dir "$run_dir"
```

成功时打印 `Serving: dsv4-flash-int8; prompt logprobs: 33 tokens`。然后运行 `abstract_algebra` 的两题 smoke：

```bash
"$MMLU_PYTHON" "$VLLM_ROOT/scripts/mmlu/run_eval.py" \
  --lm-eval "$HOME/final_test/mmlu/.venv/bin/lm-eval" \
  --harness-dir "$HOME/final_test/mmlu/lm-evaluation-harness" \
  --dataset-dir "$HOME/final_test/mmlu/cais-mmlu" \
  --task-dir "$HOME/final_test/mmlu/tasks/mmlu-local" \
  --model "$model_id" --output-dir "$run_dir" --mode smoke
```

smoke 通过后运行完整 MMLU：

```bash
"$MMLU_PYTHON" "$VLLM_ROOT/scripts/mmlu/run_eval.py" \
  --lm-eval "$HOME/final_test/mmlu/.venv/bin/lm-eval" \
  --harness-dir "$HOME/final_test/mmlu/lm-evaluation-harness" \
  --dataset-dir "$HOME/final_test/mmlu/cais-mmlu" \
  --task-dir "$HOME/final_test/mmlu/tasks/mmlu-local" \
  --model "$model_id" --output-dir "$run_dir" --mode full
```

结果保存在本节的 `run_dir`。评测完成后停止服务；只运行 smoke 时也用这组命令停止：

```bash
kill "$(cat "$run_dir/server.pid")"
while kill -0 "$(cat "$run_dir/server.pid")" 2>/dev/null; do sleep 1; done
```

## 6. 查看结果

结果都保存在本次 `run_dir`：

| 文件 | 内容 |
| --- | --- |
| `smoke/<模型名>/results_*.json`、`full/<模型名>/results_*.json` | 汇总和各学科准确率，指标为 `acc,none` |
| 同目录的 `samples_*.jsonl` | 逐题答案、选项得分和正确性 |
| `smoke.log`、`full.log` | 评测过程日志 |
| `server.log`、`server.pid` | 服务日志和进程号 |
| `models.json`、`tokenizer-info.json`、`completion-smoke.json` | 评分接口检查的原始响应 |
| `harness.commit`、`dataset.commit` | 本次评测器和数据集版本 |

### 从已保存的文件读取分数

不需要启动服务，也不需要重新发送请求。在新 Bash 会话中执行下面的命令；示例读取机器上已保存的 BF16 smoke 结果：

```bash
results_dir="$HOME/final_test/results/autocalib-bf16-20260929/smoke/dsv4-flash-bf16"
"$HOME/final_test/mmlu/.venv/bin/python" - "$results_dir" <<'PY'
import json
import sys
from pathlib import Path

for path in sorted(Path(sys.argv[1]).glob("results_*.json")):
    data = json.loads(path.read_text())
    print(f"\n文件：{path}")
    for task, metrics in data["results"].items():
        if "acc,none" in metrics:
            accuracy = metrics["acc,none"]
            print(f"{task}: {accuracy:.6f} ({accuracy:.2%})")
PY
```

示例输出为 `mmlu_abstract_algebra: 1.000000 (100.00%)`，对应这次 smoke 的两题全部答对。JSON 中 `n-samples.mmlu_abstract_algebra.effective` 为 `2`，是实际评测题数。

读取自己的完整测试时，将 `results_dir` 改为本次结果目录下的 `full/dsv4-flash-bf16` 或 `full/dsv4-flash-int8`；读取 INT8 smoke 时使用 `smoke/dsv4-flash-int8`。目录中有多次评测文件时，命令会分别打印文件名和分数。

- `acc,none` 是准确率，`0.75` 表示 `75%`。
- 完整测试看 `mmlu` 行的汇总分数；`mmlu_abstract_algebra` 等行是各学科分数。
- smoke 只跑 abstract algebra 的两题，其 `100%` 不能作为完整 MMLU 分数，也不能用于比较 BF16 和 INT8 的精度。

## 实测记录与常见问题

自动校准交付版：vLLM `b7c95b73d4`，fused_cpp `de5a940`。两种模型都使用上面的启动脚本、NUMA 4–7，未传入 planner profile；接口检查和 5-shot 两题 smoke 均通过。完整 MMLU 尚未运行。

| 模型 | 两题 smoke | 已保存的结果目录 |
| --- | --- | --- |
| BF16 | 2/2 | `/home/zhangxu/final_test/results/autocalib-bf16-20260929/` |
| W8A8 INT8 | 2/2 | `/home/zhangxu/final_test/results/autocalib-int8-20260929/` |

| 问题 | 处理 |
| --- | --- |
| 服务导入报缺少 `libnvrtc.so.13` | 在服务环境中运行 `uv pip uninstall --python "$VLLM_PYTHON" torchcodec` |
| 数据检查报告仍是 Git LFS 指针 | 重跑第 3 节的 `download_dataset.py`，再运行 `check_dataset.py` |
| 导入到旧 vLLM/fused_cpp，或调用了错误的 Python | 核对启动脚本的三个路径；服务用 `test/bin/python`，评测用 `mmlu/.venv/bin/python` |
| shared MLP 的三个用例因不足 8 核而跳过 | 旧 W8A8 内核会污染调用线程 affinity；更新到 `de5a940` 或之后的提交，并重新安装 fused_cpp |

接手时注意：两题 smoke 只验证流程；BF16/INT8 精度比较需要完整结果和相同数据版本。服务和评测器的上下文长度都是 4096，若完整评测日志出现截断，不能将结果作为未截断的 5-shot 精度使用。
