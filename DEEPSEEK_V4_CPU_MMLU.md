# DeepSeek V4 Flash CPU MMLU 评测

本文从空目录克隆 vLLM 与 AOT fused_cpp，安装服务环境和独立的 MMLU 环境，再通过 vLLM Completions 服务运行 5-shot MMLU。命令面向 `arm-codex-internal` 的 Linux AArch64、NUMA 4–7，默认在同一个 Bash shell 中依次执行。BF16 和 W8A8 INT8 分开启动，结果分别保存。

工作区另有 `bench/mmlu/bench_s.sh`，但它面向 DeepSeek-V2，写死了旧服务地址和 tokenizer 路径，还会调用 `pip` 安装依赖；不要直接拿它评测这里的 v0.28.0 服务。本流程使用 [lm-evaluation-harness 的 `local-completions` 后端](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/API_guide.md)，因为 MMLU 是需要候选答案 loglikelihood 的多选任务，Chat Completions 后端不适用。

## 1. 克隆代码与安装系统依赖

系统要求 GCC/G++ 12.3 及以上、CMake、Ninja、NUMA 开发库、jemalloc、Git、Git LFS 和 curl。`arm-codex-internal` 已有这些工具；新机器按发行版安装其中一组：

```bash
# openEuler
sudo dnf install -y gcc gcc-c++ jemalloc numactl numactl-devel \
  cmake ninja-build git git-lfs curl

# Ubuntu：仅在 Ubuntu 上执行
sudo apt-get update
sudo apt-get install -y gcc g++ libjemalloc2 libjemalloc-dev \
  numactl libnuma-dev cmake ninja-build git git-lfs curl

gcc --version
g++ --version
git lfs version
```

在测试机遇到 HTTPS 克隆的 HTTP/2 错误时，可以先运行 `git config --global http.version HTTP/1.1`。克隆本次使用的两个分支，目录和虚拟环境都放在 `final_test`：

```bash
export FINAL_TEST="$HOME/final_test"
mkdir -p "$FINAL_TEST"
git clone -b dsv4-arm-cpu-v0.28.0 --single-branch \
  https://github.com/hearth-stone/vllm-aarch64-opt.git \
  "$FINAL_TEST/vllm-aarch64-opt"
git clone -b feat/fused_cpp/dsv4_v028_aot_delivery --single-branch \
  https://github.com/hearth-stone/fused_cpp.git "$FINAL_TEST/fused_cpp"

export VLLM_ROOT="$FINAL_TEST/vllm-aarch64-opt"
export FUSED_ROOT="$FINAL_TEST/fused_cpp"
export FUSED_CPP_SRC="$FUSED_ROOT/src"
git -C "$VLLM_ROOT" rev-parse HEAD
git -C "$FUSED_ROOT" rev-parse HEAD
```

第 8 节记录了已验证的源码提交；分支更新后应先核对这里打印的提交号。

## 2. 安装 vLLM 与 fused_cpp

两份代码共用一个 Python 3.12 虚拟环境。`arm-codex-internal` 已有 `uv`；新机器先运行安装命令。环境创建完成后才设置 `VLLM_PYTHON`。[uv 安装说明](https://docs.astral.sh/uv/getting-started/installation/)

```bash
# 已安装 uv 时跳过下面两行
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"
uv python install 3.12
uv venv --python 3.12 "$FINAL_TEST/test"
source "$FINAL_TEST/test/bin/activate"
export VLLM_PYTHON="$FINAL_TEST/test/bin/python"

cd "$VLLM_ROOT"
uv pip install -r requirements/build/cpu.txt --torch-backend cpu \
  --index-strategy unsafe-best-match \
  --default-index https://mirrors.aliyun.com/pypi/simple/
uv pip install -r requirements/cpu.txt --torch-backend cpu \
  --index-strategy unsafe-best-match \
  --default-index https://mirrors.aliyun.com/pypi/simple/
VLLM_TARGET_DEVICE=cpu MAX_JOBS=16 uv pip install -e . --no-build-isolation
uv pip uninstall torchcodec

cd "$FUSED_ROOT"
FUSED_CPP_SVE_VECTOR_BITS=256 MAX_JOBS=16 \
  uv pip install -e . --no-build-isolation --no-deps
uv pip install pytest tblib
```

这里选择 256 位 SVE 构建，与测试机的运行长度相同。`torchcodec` 在测试机的 CPU requirements 安装后导入时需要缺失的 `libnvrtc.so.13`；本流程只评测文本，移除后已验证服务能启动。确认当前环境导入的是刚克隆的两份代码：

```bash
PYTHONPATH="$FUSED_CPP_SRC:$VLLM_ROOT" "$VLLM_PYTHON" -c \
  'import fused_cpp, vllm; print(fused_cpp.__file__); print(vllm.__file__)'
```

## 3. 测试关键 kernel

运行你列出的三个 fused_cpp 测试文件；`PYTHONPATH` 指向本次克隆，避免导入机器上的旧安装：

```bash
cd "$FUSED_ROOT"
PYTHONPATH="$FUSED_CPP_SRC:$VLLM_ROOT" "$VLLM_PYTHON" -m pytest -q -rs \
  tests/test_fused_moe_bf16_tiled.py \
  tests/test_deepseek_v4_attn_gemm_fused.py \
  tests/test_deepseek_v4_inv_rope_woa.py
```

vLLM 的 DeepSeek V4 CPU 专项测试需要模型配置文件。目标机已有权重时，可以接着运行：

```bash
mkdir -p "$VLLM_ROOT/model_configs/DeepSeek-V4-Flash-BF16"
cp /mnt/models/DeepSeek-V4-Flash-BF16/config.json \
  "$VLLM_ROOT/model_configs/DeepSeek-V4-Flash-BF16/config.json"
cd "$VLLM_ROOT"
VLLM_TARGET_DEVICE=cpu PYTHONPATH="$FUSED_CPP_SRC:$VLLM_ROOT" \
  "$VLLM_PYTHON" -m pytest -q -rs \
  tests/models/test_deepseek_v4_cpu_v028.py
```

## 4. 创建独立的 MMLU 环境与数据集

MMLU 评测器使用另一个 Python 3.12 环境，通过服务端 tokenizer 和 Completions 接口评分。先创建环境，再设置 `MMLU_PYTHON`；固定评测器与数据集提交，以便两种精度使用同一套题目。

```bash
export MMLU_WORKDIR="$FINAL_TEST/mmlu"
export HARNESS_DIR="$MMLU_WORKDIR/lm-evaluation-harness"
export DATASET_DIR="$MMLU_WORKDIR/cais-mmlu"
export TASK_DIR="$MMLU_WORKDIR/tasks/mmlu-local"
mkdir -p "$MMLU_WORKDIR"

git clone https://github.com/EleutherAI/lm-evaluation-harness.git "$HARNESS_DIR"
git -C "$HARNESS_DIR" checkout c1c4bea3777f73e188395264083adcf454913344
uv venv --python 3.12 "$MMLU_WORKDIR/.venv"
export MMLU_PYTHON="$MMLU_WORKDIR/.venv/bin/python"
uv pip install --python "$MMLU_PYTHON" -e "${HARNESS_DIR}[api]"

GIT_LFS_SKIP_SMUDGE=1 git clone \
  https://hf-mirror.com/datasets/cais/mmlu "$DATASET_DIR"
GIT_LFS_SKIP_SMUDGE=1 git -C "$DATASET_DIR" checkout \
  c30699e8356da336a370243923dbaf21066bb9fe
"$MMLU_PYTHON" "$VLLM_ROOT/scripts/mmlu/download_dataset.py" \
  --dataset-dir "$DATASET_DIR" \
  --hf-cli "$MMLU_WORKDIR/.venv/bin/hf"
"$MMLU_PYTHON" "$VLLM_ROOT/scripts/mmlu/check_dataset.py" \
  --dataset-dir "$DATASET_DIR"
"$MMLU_PYTHON" "$VLLM_ROOT/scripts/mmlu/prepare_tasks.py" \
  --harness-dir "$HARNESS_DIR" \
  --dataset-dir "$DATASET_DIR" \
  --task-dir "$TASK_DIR"
```

`arm-codex-internal` 无法直接连接 `huggingface.co`，上面的镜像下载方式在目标机通过了 176 个 Parquet 文件的检查；下载脚本会重试残留的 Git LFS 指针。`prepare_tasks.py` 复制 lm-eval 自带的 MMLU 配置并改为本地数据集路径。

## 5. 校准并启动 BF16 服务

当前推送的 vLLM 分支需要每个 TP rank 的 MoE profile。先为 NUMA 4–7 的 4 组 40 核生成 profile，然后直接启动 API server。两种精度共用这组 profile；仓库里名称含 `autocalib` 的启动脚本会清掉 profile 变量，因此此处不用它们。目标机还需已有 `/mnt/models/DeepSeek-V4-Flash-BF16/` 和 `/mnt/models/DeepSeek-V4-Flash-INT8/`。

```bash
export PROFILE_DIR="$FINAL_TEST/profiles/mmlu40"
PYTHONPATH="$FUSED_CPP_SRC:$VLLM_ROOT" OMP_NUM_THREADS=40 \
  "$VLLM_PYTHON" "$VLLM_ROOT/scripts/mmlu/calibrate_profiles.py" \
  --output-dir "$PROFILE_DIR" --first-cpu 160 \
  --rank-count 4 --threads-per-rank 40

export PYTHONPATH="$FUSED_CPP_SRC:$VLLM_ROOT"
export LD_PRELOAD=/usr/lib64/libjemalloc.so
export MALLOC_CONF=thp:always,oversize_threshold:2097152,background_thread:true
export GLOO_DEVICE_TRANSPORT=TCP GLOO_SOCKET_IFNAME=eno1
export VLLM_CPU_KVCACHE_SPACE=10 VLLM_TARGET_DEVICE=cpu
export VLLM_CPU_FUSED_CPP_STRICT=1
export VLLM_CPU_OMP_THREADS_BIND='160-199|200-239|240-279|280-319'
export OMP_NUM_THREADS=40 MKL_NUM_THREADS=40 NUMEXPR_MAX_THREADS=40
export OPENBLAS_NUM_THREADS=40 VECLIB_MAXIMUM_THREADS=40 GOTO_NUM_THREADS=40
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=36000
export FUSED_CPP_MOE_PLANNER_PROFILE="$PROFILE_DIR/rank{local_rank}.json"

export MODEL_ID=dsv4-flash-bf16
export RUN_DIR="$FINAL_TEST/results/bf16-$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$RUN_DIR"
cd "$VLLM_ROOT"
nohup "$VLLM_PYTHON" -m vllm.entrypoints.openai.api_server \
  --model /mnt/models/DeepSeek-V4-Flash-BF16/ \
  --host 127.0.0.1 --port 8004 \
  --gpu-memory-utilization 0.95 --max-model-len 4096 \
  --tensor-parallel-size 4 --dtype bfloat16 --trust-remote-code \
  --numa-bind --numa-bind-nodes 4 5 6 7 \
  --numa-bind-cpus 160-199 200-239 240-279 280-319 \
  --served-model-name "$MODEL_ID" \
  --enable-tokenizer-info-endpoint --no-enable-prefix-caching \
  > "$RUN_DIR/server.log" 2>&1 < /dev/null &
echo $! > "$RUN_DIR/server.pid"
```

首次加载模型可能需要数分钟；确认服务就绪：

```bash
until curl -fsS -o /dev/null http://127.0.0.1:8004/v1/models; do
  if ! kill -0 "$(cat "$RUN_DIR/server.pid")" 2>/dev/null; then
    tail -n 60 "$RUN_DIR/server.log"
    exit 1
  fi
  sleep 10
done
```

## 6. 检查评分接口并运行 MMLU

MMLU 使用 completion 端点的 `echo` 和 prompt token logprobs。普通生成请求成功，不等于这种评分请求也成功。脚本检查模型名、tokenizer 接口和 prompt logprobs，并保存原始响应：

```bash
"$MMLU_PYTHON" "$VLLM_ROOT/scripts/mmlu/check_api.py" \
  --model "$MODEL_ID" \
  --output-dir "$RUN_DIR"
```

评测器走服务端 `/tokenize`、`/detokenize` 和 `/tokenizer_info`，因此 `tokenizer_backend=remote`。服务端脚本的最大上下文长度为 4096，客户端 `max_length` 与其保持一致。先用一个学科的两道题验证端到端流程；`--limit` 仅用于调试，不能把该结果当作完整 MMLU 成绩。[lm-eval 参数及结果输出说明](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/interface.md)

```bash
"$MMLU_PYTHON" "$VLLM_ROOT/scripts/mmlu/run_eval.py" \
  --lm-eval "$MMLU_WORKDIR/.venv/bin/lm-eval" \
  --harness-dir "$HARNESS_DIR" \
  --dataset-dir "$DATASET_DIR" \
  --task-dir "$TASK_DIR" \
  --model "$MODEL_ID" \
  --output-dir "$RUN_DIR" \
  --mode smoke
```

小样本成功后可运行全部 MMLU。`--output_path` 保存汇总 JSON；`--log_samples` 保存每题的请求、响应及评分信息。本次服务测试执行了小样本，没有执行完整 MMLU。

```bash
"$MMLU_PYTHON" "$VLLM_ROOT/scripts/mmlu/run_eval.py" \
  --lm-eval "$MMLU_WORKDIR/.venv/bin/lm-eval" \
  --harness-dir "$HARNESS_DIR" \
  --dataset-dir "$DATASET_DIR" \
  --task-dir "$TASK_DIR" \
  --model "$MODEL_ID" \
  --output-dir "$RUN_DIR" \
  --mode full

ls -R "$RUN_DIR/full"
```

`$RUN_DIR/full/` 下的模型子目录包含 `results_*.json`（整体和各学科准确率）以及 `samples_*.jsonl`（逐题记录）；`$RUN_DIR` 中的 `full.log`、`server.log`、两个提交号、`models.json`、`tokenizer-info.json` 和 `completion-smoke.json` 用于复核运行配置。若日志提示上下文被截断，不要把该运行结果与更长上下文的 5-shot MMLU 结果直接比较；先增大服务端 `--max-model-len`，并给 `run_eval.py` 传相同的 `--max-length` 后重新运行。

## 7. 切换到 W8A8 INT8

BF16 评测结束后停止服务，使用相同的环境和 profile 在端口 8004 启动 W8A8 INT8。以下命令保存到新的结果目录：

```bash
kill "$(cat "$RUN_DIR/server.pid")"
while kill -0 "$(cat "$RUN_DIR/server.pid")" 2>/dev/null; do sleep 1; done

export MODEL_ID=dsv4-flash-int8
export RUN_DIR="$FINAL_TEST/results/int8-$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$RUN_DIR"
cd "$VLLM_ROOT"
nohup "$VLLM_PYTHON" -m vllm.entrypoints.openai.api_server \
  --model /mnt/models/DeepSeek-V4-Flash-INT8/ \
  --host 127.0.0.1 --port 8004 \
  --gpu-memory-utilization 0.95 --max-model-len 4096 \
  --tensor-parallel-size 4 --dtype bfloat16 --trust-remote-code \
  --quantization compressed-tensors \
  --numa-bind --numa-bind-nodes 4 5 6 7 \
  --numa-bind-cpus 160-199 200-239 240-279 280-319 \
  --served-model-name "$MODEL_ID" \
  --enable-tokenizer-info-endpoint --no-enable-prefix-caching \
  > "$RUN_DIR/server.log" 2>&1 < /dev/null &
echo $! > "$RUN_DIR/server.pid"

until curl -fsS -o /dev/null http://127.0.0.1:8004/v1/models; do
  if ! kill -0 "$(cat "$RUN_DIR/server.pid")" 2>/dev/null; then
    tail -n 60 "$RUN_DIR/server.log"
    exit 1
  fi
  sleep 10
done

"$MMLU_PYTHON" "$VLLM_ROOT/scripts/mmlu/check_api.py" \
  --model "$MODEL_ID" --output-dir "$RUN_DIR"

"$MMLU_PYTHON" "$VLLM_ROOT/scripts/mmlu/run_eval.py" \
  --lm-eval "$MMLU_WORKDIR/.venv/bin/lm-eval" \
  --harness-dir "$HARNESS_DIR" \
  --dataset-dir "$DATASET_DIR" \
  --task-dir "$TASK_DIR" \
  --model "$MODEL_ID" \
  --output-dir "$RUN_DIR" \
  --mode smoke
```

INT8 小样本成功后，按第 6 节的完整评测命令运行 `--mode full`；当前 `MODEL_ID` 和 `RUN_DIR` 已指向 INT8。完成所有评测后，运行 `kill "$(cat "$RUN_DIR/server.pid")"` 停止服务。

## 8. `arm-codex-internal` 实测记录

2026-09-29 在 `$HOME/final_test` 从空目录安装了 vLLM、AOT fused_cpp、服务虚拟环境和独立 MMLU 环境。vLLM 使用 `549a522c9dca299164ea48c4c7fa533081f7b795`，fused_cpp 使用 `6f8bfbccddf58f0f9d673704ad39c7903ed29c64`。完整 fused_cpp 测试运行 `PYTHONPATH="$FUSED_CPP_SRC:$VLLM_ROOT" "$VLLM_PYTHON" -m pytest -q -rs` 得到 `1577 passed, 4 skipped`；vLLM 专项测试 `tests/models/test_deepseek_v4_cpu_v028.py` 得到 `30 passed`。这组结果包含第 3 节列出的三个内核测试文件。

`check_dataset.py` 验证了 176 个 Parquet 文件，离线加载 `abstract_algebra` 得到 `dev=5`、`test=100`。lm-eval 固定在 `c1c4bea3777f73e188395264083adcf454913344`，数据集固定在 `c30699e8356da336a370243923dbaf21066bb9fe`。

| 模型 | 评分接口 | 5-shot 两题小样本 | 结果目录 |
| --- | --- | --- | --- |
| BF16 | prompt logprobs 33 tokens | 2/2，结果 JSON 与逐题 JSONL 已保存 | `/home/zhangxu/final_test/results/bf16-20260929T163733Z/` |
| W8A8 INT8 | prompt logprobs 33 tokens | 2/2，结果 JSON 与逐题 JSONL 已保存 | `/home/zhangxu/final_test/results/int8-20260929T164856Z/` |

两题结果只证明流程跑通，不能用于比较模型精度。完整 MMLU 尚未运行。两个测试服务已停止，结果文件保留在上述目录。
