# DeepSeek V4 Flash CPU MMLU 评测

本文在 `arm-codex-internal` 上从零建立独立的 MMLU 评测环境，通过本仓库的 vLLM OpenAI Completions 服务运行 5-shot MMLU，并保存汇总结果、逐题记录和运行日志。命令默认在同一个 Bash shell 中依次执行。BF16 和 W8A8 INT8 使用同一套评测步骤，分别运行并保存到不同目录。

工作区另有 `bench/mmlu/bench_s.sh`，但它面向 DeepSeek-V2，写死了旧服务地址和 tokenizer 路径，还会调用 `pip` 安装依赖；不要直接拿它评测这里的 v0.28.0 服务。本流程使用 [lm-evaluation-harness 的 `local-completions` 后端](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/API_guide.md)，因为 MMLU 是需要候选答案 loglikelihood 的多选任务，Chat Completions 后端不适用。

## 1. 准备工具和目录

目标机需要 `git`、`git-lfs`、`git-xet`、`uv` 和 `curl`。`arm-codex-internal` 当前已有这些工具。若在新机器上安装，请参阅 [uv 安装说明](https://docs.astral.sh/uv/getting-started/installation/)和 [Git Xet 安装说明](https://huggingface.co/docs/hub/xet/using-xet-storage)。先确认：

```bash
uv --version
git lfs version
git xet --version
git xet install

export MMLU_WORKDIR="$HOME/dsv4-mmlu"
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

本仓库所用 MMLU task 的数据源是 [`cais/mmlu`](https://huggingface.co/datasets/cais/mmlu)。它是包含各学科 `dev` 与 `test` Parquet 文件的 Hugging Face 数据集仓库；Git 克隆需要 Git Xet。[Hugging Face 克隆说明](https://huggingface.co/docs/hub/datasets-downloading)

```bash
git clone https://huggingface.co/datasets/cais/mmlu "$DATASET_DIR"
git -C "$DATASET_DIR" rev-parse HEAD
test -s "$DATASET_DIR/abstract_algebra/test-00000-of-00001.parquet"
```

lm-eval 内置 MMLU task 默认读取在线的 `cais/mmlu`。复制其 task 配置，只把数据集路径改为刚克隆的本地目录；`--include_path` 会优先使用这些同名配置。这样评测实际读取的是本地副本，且保留原有的题目模板、5-shot `dev` 样例与准确率计算方式。

```bash
"$MMLU_PYTHON" - <<'PY'
import json
import os
import shutil
from pathlib import Path

source = Path(os.environ["HARNESS_DIR"]) / "lm_eval/tasks/mmlu/default"
target = Path(os.environ["TASK_DIR"])
shutil.copytree(source, target, dirs_exist_ok=True)
template = target / "_default_template_yaml"
old = "dataset_path: cais/mmlu"
content = template.read_text()
assert content.count(old) == 1
template.write_text(
    content.replace(old, f"dataset_path: {json.dumps(os.environ['DATASET_DIR'])}")
)
print(template.read_text().splitlines()[0])
PY

HF_DATASETS_OFFLINE=1 "$MMLU_PYTHON" - <<'PY'
import os
from datasets import load_dataset

data = load_dataset(os.environ["DATASET_DIR"], "abstract_algebra")
print({split: len(rows) for split, rows in data.items()})
assert data["dev"] and data["test"]
PY
```

## 4. 启动 vLLM 服务

下面以 [BF16 启动脚本](run_numa_4567_bf16_autocalib.sh)为例。三项路径必须由调用方提供；把它们改成目标机的实际位置。启动脚本固定使用 NUMA 4、5、6、7，每个 TP rank 40 核。`--enable-tokenizer-info-endpoint` 让评测器能够通过服务端完成 tokenization；`--served-model-name` 固定请求中的模型名。

```bash
export VLLM_ROOT=/home/zhangxu/codex/vllm-aarch64-v0.28.0
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

W8A8 INT8 使用[对应的启动脚本](run_numa_4567_w8a8_autocalib.sh)和独立结果目录；不要让两个服务同时占用 `127.0.0.1:8004`：

```bash
export MODEL_ID=dsv4-flash-int8
export RUN_DIR="$MMLU_WORKDIR/results/int8-$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$RUN_DIR"
nohup "$VLLM_ROOT/run_numa_4567_w8a8_autocalib.sh" \
  --served-model-name "$MODEL_ID" \
  --enable-tokenizer-info-endpoint \
  --no-enable-prefix-caching \
  > "$RUN_DIR/server.log" 2>&1 < /dev/null &
echo $! > "$RUN_DIR/server.pid"
```

首次启动会加载模型并在 `/tmp` 自动生成各 rank 的 MoE 校准文件，可能需要数分钟。确认服务就绪并保存服务端模型信息：

```bash
until curl -fsS http://127.0.0.1:8004/v1/models > "$RUN_DIR/models.json"; do
  if ! kill -0 "$(cat "$RUN_DIR/server.pid")" 2>/dev/null; then
    tail -n 60 "$RUN_DIR/server.log"
    exit 1
  fi
  sleep 10
done
curl -fsS -o /dev/null http://127.0.0.1:8004/tokenizer_info

"$MMLU_PYTHON" - <<'PY'
import json
import os
from pathlib import Path

models = json.loads((Path(os.environ["RUN_DIR"]) / "models.json").read_text())
assert os.environ["MODEL_ID"] in {model["id"] for model in models["data"]}
print("Serving:", os.environ["MODEL_ID"])
PY
```

## 5. 先检查一次 MMLU 所需的请求形式

MMLU 使用 completion 端点的 `echo` 和 prompt token logprobs。普通生成请求成功，不等于这种评分请求也成功。保存一次原始响应并检查返回的 prompt logprobs：

```bash
curl -fsS http://127.0.0.1:8004/v1/completions \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MODEL_ID\",\"prompt\":\"Question: 1 + 1 = ?\\nA. 1\\nB. 2\\nC. 3\\nD. 4\\nAnswer: B\",\"max_tokens\":1,\"temperature\":0,\"echo\":true,\"logprobs\":1}" \
  > "$RUN_DIR/completion-smoke.json"

"$MMLU_PYTHON" - <<'PY'
import json
import os
from pathlib import Path

response = json.loads(
    (Path(os.environ["RUN_DIR"]) / "completion-smoke.json").read_text()
)
logprobs = response["choices"][0]["logprobs"]["token_logprobs"]
assert len(logprobs) > 2 and any(value is not None for value in logprobs[:-1])
print(f"Prompt logprobs available: {len(logprobs)} tokens")
PY
```

## 6. 小样本验证，再跑完整 MMLU

评测器走服务端 `/tokenize`、`/detokenize` 和 `/tokenizer_info`，因此 `tokenizer_backend=remote`。服务端脚本的最大上下文长度为 4096，客户端 `max_length` 与其保持一致。先用一个学科的两道题验证端到端流程；`--limit` 仅用于调试，不能把该结果当作完整 MMLU 成绩。[lm-eval 参数及结果输出说明](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/interface.md)

```bash
export HF_DATASETS_OFFLINE=1
export MODEL_ARGS="model=$MODEL_ID,base_url=http://127.0.0.1:8004/v1/completions,tokenizer_backend=remote,num_concurrent=1,timeout=3600,max_retries=3,max_length=4096"
set -o pipefail

"$MMLU_WORKDIR/.venv/bin/lm-eval" \
  --model local-completions \
  --model_args "$MODEL_ARGS" \
  --tasks mmlu_abstract_algebra \
  --include_path "$TASK_DIR" \
  --num_fewshot 5 --batch_size 1 --limit 2 \
  --output_path "$RUN_DIR/smoke" --log_samples \
  2>&1 | tee "$RUN_DIR/smoke.log"
```

小样本成功后运行全部 MMLU。`--output_path` 保存汇总 JSON；`--log_samples` 保存每题的请求、响应及评分信息。

```bash
git -C "$HARNESS_DIR" rev-parse HEAD > "$RUN_DIR/harness.commit"
git -C "$DATASET_DIR" rev-parse HEAD > "$RUN_DIR/dataset.commit"

"$MMLU_WORKDIR/.venv/bin/lm-eval" \
  --model local-completions \
  --model_args "$MODEL_ARGS" \
  --tasks mmlu \
  --include_path "$TASK_DIR" \
  --num_fewshot 5 --batch_size 1 \
  --output_path "$RUN_DIR/full" --log_samples \
  2>&1 | tee "$RUN_DIR/full.log"

ls -R "$RUN_DIR/full"
```

`$RUN_DIR/full/` 下的模型子目录包含 `results_*.json`（整体和各学科准确率）以及 `samples_*.jsonl`（逐题记录）；`$RUN_DIR` 中的 `full.log`、`server.log`、两个提交号及 `models.json` 用于复核运行配置。若日志提示上下文被截断，不要把该运行结果与更长上下文的 5-shot MMLU 结果直接比较；先增大服务端 `--max-model-len` 并同步增大客户端 `max_length`，然后重新运行。
