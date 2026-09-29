# DeepSeek V4 Flash CPU MMLU 评测

本文在 `arm-codex-internal` 上从零建立独立的 MMLU 评测环境，通过本仓库的 vLLM OpenAI Completions 服务运行 5-shot MMLU，并保存汇总结果、逐题记录和运行日志。命令默认在同一个 Bash shell 中依次执行。BF16 和 W8A8 INT8 使用同一套评测步骤，分别运行并保存到不同目录。

工作区另有 `bench/mmlu/bench_s.sh`，但它面向 DeepSeek-V2，写死了旧服务地址和 tokenizer 路径，还会调用 `pip` 安装依赖；不要直接拿它评测这里的 v0.28.0 服务。本流程使用 [lm-evaluation-harness 的 `local-completions` 后端](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/API_guide.md)，因为 MMLU 是需要候选答案 loglikelihood 的多选任务，Chat Completions 后端不适用。

## 1. 准备工具和目录

目标机需要 `git`、`git-lfs`、`uv` 和 `curl`。`arm-codex-internal` 当前已有这些工具。若在新机器上安装 `uv`，请参阅 [安装说明](https://docs.astral.sh/uv/getting-started/installation/)。先确认：

```bash
uv --version
git lfs version

export MMLU_WORKDIR="$HOME/dsv4-mmlu"
export VLLM_ROOT=/home/zhangxu/codex/vllm-aarch64-v0.28.0
export HARNESS_DIR="$MMLU_WORKDIR/lm-evaluation-harness"
export DATASET_DIR="$MMLU_WORKDIR/cais-mmlu"
export TASK_DIR="$MMLU_WORKDIR/tasks/mmlu-local"
mkdir -p "$MMLU_WORKDIR"
```

## 2. 用 uv 创建独立评测环境

克隆 lm-evaluation-harness 并固定到本文检查过的提交。先让 `uv` 安装 Python 3.12 并创建虚拟环境，再设置 `MMLU_PYTHON`；这个路径在创建环境之前并不存在。`[api]` extra 包含 HTTP 请求所需依赖；评测进程通过服务端 tokenizer 接口工作，无需在评测环境中安装 vLLM 或加载模型权重。[uv 虚拟环境与安装文档](https://docs.astral.sh/uv/pip/environments/)

```bash
git clone https://github.com/EleutherAI/lm-evaluation-harness.git "$HARNESS_DIR"
git -C "$HARNESS_DIR" checkout c1c4bea3777f73e188395264083adcf454913344

uv python install 3.12
uv venv --python 3.12 "$MMLU_WORKDIR/.venv"
export MMLU_PYTHON="$MMLU_WORKDIR/.venv/bin/python"
test -x "$MMLU_PYTHON"
"$MMLU_PYTHON" --version
uv pip install --python "$MMLU_PYTHON" -e "${HARNESS_DIR}[api]"
"$MMLU_WORKDIR/.venv/bin/lm-eval" --help > /dev/null
```

## 3. 克隆 MMLU 数据集并指向本地副本

本仓库所用 MMLU task 的数据源是 [`cais/mmlu`](https://huggingface.co/datasets/cais/mmlu)，包含各学科的 `dev` 与 `test` Parquet 文件。2026-09-29 在 `arm-codex-internal` 上，`huggingface.co` 无法连接；以下使用实际测试可用的 `hf-mirror.com`。先克隆 Git 元数据且跳过 Git LFS 自动下载，再用已安装的 `hf` CLI 下载同一提交的全部文件。镜像的 Xet API 返回 401，因此下载脚本禁用 Xet，走普通 HTTP；若 CLI 因已有 Git LFS 指针而跳过个别文件，脚本会单独重试。

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone https://hf-mirror.com/datasets/cais/mmlu "$DATASET_DIR"
"$MMLU_PYTHON" "$VLLM_ROOT/scripts/mmlu/download_dataset.py" \
  --dataset-dir "$DATASET_DIR" \
  --hf-cli "$MMLU_WORKDIR/.venv/bin/hf"

"$MMLU_PYTHON" "$VLLM_ROOT/scripts/mmlu/check_dataset.py" \
  --dataset-dir "$DATASET_DIR"
```

lm-eval 内置 MMLU task 默认读取在线的 `cais/mmlu`。复制其 task 配置，只把数据集路径改为刚克隆的本地目录；`--include_path` 会优先使用这些同名配置。这样评测实际读取的是本地副本，且保留原有的题目模板、5-shot `dev` 样例与准确率计算方式。

```bash
"$MMLU_PYTHON" "$VLLM_ROOT/scripts/mmlu/prepare_tasks.py" \
  --harness-dir "$HARNESS_DIR" \
  --dataset-dir "$DATASET_DIR" \
  --task-dir "$TASK_DIR"
```

## 4. 启动 vLLM 服务

下面以 [BF16 启动脚本](run_numa_4567_bf16_autocalib.sh)为例。三项路径必须由调用方提供；把它们改成目标机的实际位置。启动脚本固定使用 NUMA 4、5、6、7，每个 TP rank 40 核。`--enable-tokenizer-info-endpoint` 让评测器能够通过服务端完成 tokenization；`--served-model-name` 固定请求中的模型名。

```bash
export VLLM_PYTHON="$VLLM_ROOT/.venv/bin/python"
export FUSED_CPP_SRC=/home/zhangxu/codex/fused_cpp-dsv4-v028-delivery/src
export MODEL_ID=dsv4-flash-bf16
export RUN_DIR="$MMLU_WORKDIR/results/bf16-$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$RUN_DIR"

nohup "$VLLM_ROOT/run_numa_4567_bf16_autocalib.sh" \
  --served-model-name "$MODEL_ID" \
  --enable-tokenizer-info-endpoint \
  --no-enable-prefix-caching \
  > "$RUN_DIR/server.log" 2>&1 < /dev/null &
echo $! > "$RUN_DIR/server.pid"
```

首次启动会加载模型并在 `/tmp` 自动生成各 rank 的 MoE 校准文件，可能需要数分钟。先确认服务就绪：

```bash
until curl -fsS -o /dev/null http://127.0.0.1:8004/v1/models; do
  if ! kill -0 "$(cat "$RUN_DIR/server.pid")" 2>/dev/null; then
    tail -n 60 "$RUN_DIR/server.log"
    exit 1
  fi
  sleep 10
done
```

## 5. 先检查一次 MMLU 所需的请求形式

MMLU 使用 completion 端点的 `echo` 和 prompt token logprobs。普通生成请求成功，不等于这种评分请求也成功。脚本检查模型名、tokenizer 接口和 prompt logprobs，并保存原始响应：

```bash
"$MMLU_PYTHON" "$VLLM_ROOT/scripts/mmlu/check_api.py" \
  --model "$MODEL_ID" \
  --output-dir "$RUN_DIR"
```

## 6. 小样本验证，再跑完整 MMLU

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

BF16 评测结束后停止服务，再用[对应的启动脚本](run_numa_4567_w8a8_autocalib.sh)在同一端口启动 W8A8 INT8。以下命令保存到新的结果目录：

```bash
kill "$(cat "$RUN_DIR/server.pid")"
while kill -0 "$(cat "$RUN_DIR/server.pid")" 2>/dev/null; do sleep 1; done

export MODEL_ID=dsv4-flash-int8
export RUN_DIR="$MMLU_WORKDIR/results/int8-$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$RUN_DIR"
nohup "$VLLM_ROOT/run_numa_4567_w8a8_autocalib.sh" \
  --served-model-name "$MODEL_ID" \
  --enable-tokenizer-info-endpoint \
  --no-enable-prefix-caching \
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

INT8 小样本成功后，按第 6 节的完整评测命令运行 `--mode full`；当前 `MODEL_ID` 和 `RUN_DIR` 已指向 INT8。

## 8. `arm-codex-internal` 实测记录

2026-09-29 按上述流程创建了 Python 3.12 的独立 `uv` 环境，从空目录克隆并下载了数据集。`check_dataset.py` 验证了 176 个 Parquet 文件，离线加载 `abstract_algebra` 得到 `dev=5`、`test=100`。lm-eval 固定在 `c1c4bea3777f73e188395264083adcf454913344`，数据集固定在 `c30699e8356da336a370243923dbaf21066bb9fe`。

| 模型 | 评分接口 | 5-shot 两题小样本 | 结果目录 |
| --- | --- | --- | --- |
| BF16 | prompt logprobs 33 tokens | 2/2，结果 JSON 与逐题 JSONL 已保存 | `/home/zhangxu/dsv4-mmlu/results/bf16-20260929T094329Z/` |
| W8A8 INT8 | prompt logprobs 33 tokens | 2/2，结果 JSON 与逐题 JSONL 已保存 | `/home/zhangxu/dsv4-mmlu/results/int8-20260929T095730Z/` |

两题结果只证明流程跑通，不能用于比较模型精度。完整 MMLU 尚未运行。两个测试服务已停止，结果文件保留在上述目录。
