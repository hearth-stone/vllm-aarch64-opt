<!-- Copyright (c) 2026 Huawei Technologies Co., Ltd. -->

# DeepSeek V4：100 条中文 2K 输入及串行性能测试

`zh2048-public-100.jsonl` 是机器上原有的 100 条中文输入，原样纳入仓库。
100 个 case ID 和 token 序列均不重复，每条 `prompt_token_ids` 恰好包含
2048 个 DeepSeek V4 token。数据用于不同输入的 prefill 性能测试，不包含问题、
标准答案或 chat framing，不是回答准确性评测集。

## 数据来源

文本截自全国人大国家法律法规数据库的 11 份法律、地方性法规及司法解释。
每条记录保留原始标题、`source_url`、截取的 token 起点、文本和 token ID。
多个用例可能来自同一份文件的不同片段，不能当作 100 份独立文档。
`source_sha256` 是原数据已有的来源信息，本次导入未重新计算。

这些官方法律法规及司法性质文件属于《中华人民共和国著作权法》第五条第一项
列出的不适用范围，参见[国家版权局发布的法律正文](https://www.ncac.gov.cn/xxfb/flfg/flfg_532/202103/t20210309_50530.html)。
本目录保留原始文本来源；代码的 Apache-2.0 标记不替代源文本的适用规则。

## 请求服务

在 Arm-codex-internal 上测试已有 Flash BF16 prepack cache 时，先将
`scripts/serve_deepseek_v4_cpu.sh` 中的模型选择为
`/mnt/models/DeepSeek-V4-Flash-BF16`，然后在一个终端启动：

```bash
nice -n -20 bash /home/zhangxu/final_test/vllm-aarch64-opt/scripts/serve_deepseek_v4_cpu.sh
```

API、engine、rank 和新建线程继承 NI=-20。当前账户已验证可以设置该优先级。
启动脚本直接读取仓库内的
[TP4 固定校准表](../../profiles/deepseek_v4/arm-codex-tp4-80cores/README.md)，
路径为 `$vllm_root/benchmarks/profiles/deepseek_v4/arm-codex-tp4-80cores/rank{tp_rank}.json`，
并配置：

```bash
# Flash BF16 TP4 的已有专家权重预打包缓存。
export VLLM_CPU_MOE_PREPACKED_DIR=/mnt/models/.cache/zhangxu/DeepSeek-V4-Flash-BF16-moe-prepacked-v028-482e03e
# 加载缓存时提前触碰权重页。
export VLLM_CPU_MOE_PREPACK_PREFAULT=1
```

cache 版本标记为 `v028-482e03e`，它不是当前 fused_cpp 的 Git 提交号。
INT8 和 Pro 不复用这份 BF16 cache。服务 `/health` 返回 200 后，在另一个终端
先测少量输入：

```bash
cd /home/zhangxu/final_test/vllm-aarch64-opt
source /home/zhangxu/final_test/test/bin/activate
python scripts/benchmark_deepseek_v4_2k.py \
    --requests 5 --interval-seconds 5 --seed 42 --warmup 3 \
    --prepack-cache-version v028-482e03e \
    --output-dir /home/zhangxu/final_test/perf-2k-prepacked-$(date +%Y%m%d-%H%M%S)
```

`--prepack-cache-version` 将调用者提供的版本保存到 `requests.json` 的
`protocol.prepack_cache_version`，不启用或检查服务端缓存。
cache 和 prefault 必须在启动服务时设置；请求脚本的环境变量不会改变已经运行的服务。

在 vLLM 仓库根目录执行，先启动模型服务并确认 `/health` 返回 200：

```bash
source /home/zhangxu/final_test/test/bin/activate

python scripts/benchmark_deepseek_v4_2k.py \
    --requests 100 --interval-seconds 5 --seed 0 --warmup 3 \
    --output-dir /home/zhangxu/final_test/perf-2k100-$(date +%Y%m%d-%H%M%S)
```

默认服务地址为 `http://127.0.0.1:8018`，模型 ID 为 `deepseek-v4`。
可用 `--base-url` 和 `--model` 指定其他服务地址、模型名。脚本只依赖 Python
标准库，不需要 PyTorch、fused_cpp 或服务日志目录。

| 参数 | 含义 | 默认值 |
| --- | --- | --- |
| `--requests` | 本轮测量的不同输入条数，范围 1–100 | 100 |
| `--interval-seconds` | 每条响应结束后等待秒数，0 表示连续串行发送 | 5 |
| `--seed` | 随机打乱输入的种子；同一值可复现顺序 | 0 |
| `--warmup` | 测量前重复第一条选中输入的预热次数，不计入统计 | 3 |
| `--timeout-seconds` | 单次 HTTP 请求超时秒数 | 240 |
| `--prepack-cache-version` | 保存服务使用的 cache 版本标记，不启用或校验缓存 | 未设置 |
| `--dataset` | 输入 JSONL 文件，默认自动定位本目录的数据 | 仓库内数据 |
| `--output-dir` | 新建结果目录，不覆盖已有目录 | 请求模式必填 |
| `--analyze` | 离线读取已有 `requests.json` 重算统计，不发送请求 | 未设置 |

例如只测随机选出的 60 条，间隔 10 秒：

```bash
python scripts/benchmark_deepseek_v4_2k.py \
    --requests 60 --interval-seconds 10 --seed 42 --warmup 3 \
    --output-dir /home/zhangxu/final_test/perf-2k60-$(date +%Y%m%d-%H%M%S)
```

脚本先打乱整份数据，再取前 N 条，不重复采样；逐条串行发送，前一条完成并
等待指定间隔后才发下一条。预热期间也使用该间隔，最后一条之后不等待。
每次调用 `/v1/completions`，直接提交 token ID，`temperature=0`、
`max_tokens=1`、`ignore_eos=true`、非流式返回。不添加 chat 模板，不请求 logprobs。
BF16、INT8 和原始 FP8 在线转换服务均可使用；模型 tokenizer 必须与数据中的
DeepSeek V4 token ID 兼容。

E2E 从发起 HTTP 请求前计时，到完整响应体读取完毕为止，包含网络、服务排队、
prefill 和一个输出 token 的生成；不包含请求间隔、文件读写和响应 JSON 解析。
它不是纯内核时间。对比不同运行时，请保持条数、顺序种子、间隔、预热和服务
参数相同，并关闭 prefix caching 以避免复用前缀计算。

## 保存结果及离线分析

每条请求完成后，脚本更新两个标准 JSON 文件；先写临时文件再替换，保留已完成记录：

- `requests.json`：`protocol` 中保存参数和实际 case 顺序；`warmups` 保存预热；
  `requests` 保存正式请求。每条包含 case ID、标题、完整请求 payload、完整响应、
  `start_unix`/`end_unix` 时间戳、`e2e_s` 秒数、成功状态和错误（如有）。
- `summary.json`：数量、完成状态及 `avg_s`、`p50_s`、`p90_s`、`p95_s`、
  `p99_s`、`min_s`、`max_s`，每完成一条也会更新。

离线读取本脚本生成的结果文件，仅打印重新计算的统计：

```bash
python scripts/benchmark_deepseek_v4_2k.py \
    --analyze /home/zhangxu/final_test/perf-2k100-本次时间/requests.json
```

需要另存汇总时增加一个新目录：

```bash
python scripts/benchmark_deepseek_v4_2k.py \
    --analyze /home/zhangxu/final_test/perf-2k100-本次时间/requests.json \
    --output-dir /home/zhangxu/final_test/perf-2k100-recomputed
```

百分位使用排序后的成功样本线性插值，位置为 `(N-1) × q`；P50 即中位数。
预热、间隔和失败请求不计入延迟统计。只有一个成功样本时所有百分位均为该值；
没有成功样本时统计值为 `null`。

HTTP 错误、超时或返回 token 数不符时，记录失败并停止，不重试，退出码为 1；
完整完成时为 0。部分运行仍保存记录和汇总，`complete=false`，明确区分
`requested` 与实际 `attempted`。离线分析会保留这个不完整状态；分析模式退出码
0 只表示文件分析成功。此脚本验证 HTTP 成功和 token 数，不判定回答语义正确性。
