<!-- Copyright (c) 2026 Huawei Technologies Co., Ltd. -->

# 已启动 API 的 CPU 性能测试

在 `arm-codex-internal` 上执行，API 已启动。脚本只使用 Python 标准库，不加载模型。

使用已有中文 2K 数据集：精确 2048 个输入 token，预热 1 次，正式请求 20 次，串行运行，每次完成后等待 10 秒再发送下一次（预热后也等待 10 秒）。默认输出 1 个 token，适合比较 prefill 延迟。

```bash
VLLM_ROOT="$HOME/final_test/vllm-aarch64-opt"
VLLM_PYTHON="$HOME/final_test/test/bin/python"

# 查看服务的模型 ID；下面 --model 必须与返回的 id 一致。
curl -s http://127.0.0.1:8004/v1/models

"$VLLM_PYTHON" "$VLLM_ROOT/scripts/benchmark_cpu_api.py" \
  --base-url http://127.0.0.1:8004 \
  --model dsv4-flash-bf16 \
  --cases /mnt/models/.cache/zhangxu/benchmarks/fused-attn80-continuous30-20260821/cases31.jsonl \
  --output-dir "$HOME/final_test/results/perf-bf16-$(date -u +%Y%m%dT%H%M%SZ)"
```

INT8 使用相同命令，将模型 ID 改为实际的 INT8 ID，输出目录改为 `perf-int8-...`。其他端口修改 `--base-url`。

结果目录：

- `summary.json`：正式请求的平均、中位数、P95 延迟及输入 tokens/s。
- `requests.jsonl`：每次请求的延迟、实际输入/输出 token 数和完整响应。
- `warmup.json`：预热结果，不计入汇总。
- `config.json`：本次参数与数据集路径。

延迟从发送请求前计时到完整响应读取结束，包含 HTTP、排队和输出生成。输入 tokens/s 为总输入 token 数除以正式请求耗时之和，不包含 10 秒等待时间，不等同于服务总吞吐。非流式请求不报告 TTFT 或 TPOT。

比较 BF16/INT8 时使用同一数据集、输出长度和服务配置；服务开启 prefix caching 会影响 prefill 结果，比较时两边均关闭。需要测更长输出可添加 `--max-tokens 128`，此时延迟包含 decode；`ignore_eos=true` 固定请求输出长度。
