# Arm-codex TP4 MoE 固定校准表

这四份 JSON 是 Arm-codex-internal 上已经用于 Flash BF16、Flash INT8 和 Pro
测试的固定校准表，原路径为
`/home/zhangxu/final_test/cpu_runtime/profiles/rank{tp_rank}.json`。
它们采用相同的 rank0 校准数值，保留各 rank 的机器标识、来源和拓扑信息，
本次纳入仓库没有重新校准或修改数值。

适用配置为 TP4，每个 rank 80 核，绑定分别为 `0-79`、`80-159`、
`160-239`、`240-319`。这是该机器的固定测试配置；不能直接当作其他机器或
其他核数的校准结果。

`scripts/serve_deepseek_v4_cpu.sh` 通过下面的路径加载：

```bash
# 使用仓库内的固定校准表，{tp_rank} 由 MoE 后端替换为实际 TP rank。
export FUSED_CPP_MOE_PLANNER_PROFILE="$vllm_root/benchmarks/profiles/deepseek_v4/arm-codex-tp4-80cores/rank{tp_rank}.json"
```

HTTP 性能测试脚本请求已经启动的服务，校准表必须在启动服务时选择。
更改路径后需要重启服务才生效。
