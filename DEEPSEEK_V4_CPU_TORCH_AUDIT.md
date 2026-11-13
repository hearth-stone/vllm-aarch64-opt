# DeepSeek V4 ARM CPU Torch fallback 审计

审计日期：2026-09-07。基于 `dsv4-arm-cpu-v0.28.0` 的
`31e4e4e3df` 及当前工作区改动。本文是本地实现审计，不是已完成的
ARM 真实权重端到端验证，也不是上游 PR 的就绪声明。

## 结论与基线范围

当前有较完整的 attention、indexer、compressor 和 mHC Torch 参考代码，
但尚不存在一个能保证整个 DeepSeek V4 模型走 Torch 的显式模式。
`VLLM_CPU_FUSED_CPP_STRICT=0` 仅允许 fallback，仍会优先准备和使用
`fused_cpp`；普通 CPU MoE fallback 也依赖自定义 kernel。

建议第一版参考模式使用 BF16 权重、现有 ARM 可用的 vLLM 调度及通信，
将模型特有计算切换为 Torch；先验证 prefill/decode，暂不以量化或 MTP
作为首版验收条件。已有实验文档记录了 BF16 模型，但本次未连接服务器
核实权重可用性、内存余量或运行结果。TP 取决于实际模型内存需求。

## 路径清单

| 模块 | 已有实现与入口 | 缺口或处理要求 |
| --- | --- | --- |
| 模型与 metadata | `cpu/model.py`、`cpu/sparse_mla.py`，Torch slot mapping、CPU cache specs | 可复用；需真实调度下覆盖分块 prefill、decode、多个请求和分页边界 |
| attention 输入 | `attention.py::_input_projections`、`_prepare_q_kv`，Torch RMSNorm/RoPE/cache write | `process_weights_after_loading` 会尝试 fused 权重准备；显式模式必须在准备前分流 |
| compressor | `ops.py::save_compressor_states`、`compress_and_store` | 本次修复 cache 未写回；仍需 C4 overlap、C128 和跨 prefill/decode 的数值测试 |
| indexer | `attention.py::DeepseekV4CPUIndexer.forward`，逐 token、逐 head 的 Torch 打分和 top-k | 有参考实现；需验证压缩 cache 与 causal 可见范围，不应以性能作为首版通过条件 |
| sparse MLA | `ops.py::sparse_mla_reference`，Torch 打分、sink 分母和加权求和 | `_run_fused_prefill`、`_run_fused_sparse` 会先尝试 fused；需要明确旁路 |
| 输出投影 | `attention.py::_o_proj`，Torch inverse RoPE 和 grouped einsum | 依赖原始 `wo_a` 权重；`wo_b` 仍有 fused linear/INT8 路径 |
| mHC | `mhc.py::mhc_pre`、`mhc_post`、`mhc_fused_post_pre`、`hc_head` | 参考公式已存在；post-load 仍自动尝试 fused mHC，需显式禁止准备 |
| 普通 linear | `layers/utils.py::dispatch_cpu_unquantized_gemm` 最终可回退 `F.linear` | 自动 dispatch 不保证 Torch；部分投影标记要求/启用 fused，需限定模型范围的选择机制 |
| router logits | `cpu/model.py::_configure_cpu_router_gate` 标记 fused BF16 GEMM、FP32 输出 | 需定义 Torch 的乘法/累加/输出精度；简单 BF16 输出再转 FP32 不能当作等价实现 |
| sqrt-softplus/hash 路由 | `_custom_ops.py::topk_hash_softplus_sqrt` 已有 CPU Torch fallback | 目前以扩展符号是否存在作为条件；显式模式需保证选择 Torch，并验证 hash、bias、scaling 和无效 token |
| BF16 routed MoE | `oracle/unquantized.py` 在 ARM 优先 fused_cpp，再 CPU；`CPUUnquantizedExperts` 调用 prepack 和 `cpu_fused_moe` | 没有完整 Torch experts 执行后端；应优先复用 modular router，再接 Torch experts，避免重写模型层路由 |
| shared experts | `cpu/model.py::DeepseekV4MLP`，模型检测 kernel 是否融合 shared experts | Torch routed backend 不融合 shared 时必须保留单独执行，并验证缩放及 TP reduction 次数 |
| INT8 linear/MoE | `scaled_mm/cpu.py`、`fused_cpp_cpu_moe.py` | joint-owned INT8 linear 跳过普通后处理，依赖后续 fused prepare；不能仅禁止 fused 就声明支持 W8A8 |
| FP8/FP4 权重 | `fp8_requant.py` 有分块 FP8 转 INT8/BF16 工具，模型按量化配置加载 | 不等于官方所有混合精度权重均有 Torch 执行路径；需另行确定转换范围、scale 布局及内存需求 |
| loader/prepack | `default_loader.py` 依据 `VLLM_CPU_MOE_PREPACKED_DIR` 跳过原始 expert 权重和 scale | Torch 模式不能使用 fused prepack 代替原始权重，需提前拒绝冲突配置 |
| MTP | `cpu/mtp.py` 使用 CPU decoder | 继承上述依赖；作为后续独立验收项 |

上表 `cpu/` 简写为 `vllm/models/deepseek_v4/cpu/`；
`layers/` 为 `vllm/model_executor/layers/`。

## 本次修复

`save_compressor_states` 对 `state_cache[blocks, offsets]` 进行高级索引，
得到的是副本。原代码只修改这个副本，后续 compression 读取不到写入的
KV 和加上 APE 的 scores。现已显式写回原始 cache。

在已有 `tests/models/test_deepseek_v4_cpu_v028.py` 中增加一个行为测试：
验证非连续、带 page padding 的 cache 实际发生写入，负 slot 被忽略，
未使用的存储和 padding 保持原值。

测试设计：模块负责持久化 compressor state；输入是 KV、scores、APE、
位置和物理 slot，输出是原地更新的 cache；防止高级索引副本写入丢失；
最小验证层级为单元测试，无需加载模型。

已在本机 `.venv`（Python 3.12、Torch 2.13.0）验证：

- 修改前原有套件：24 passed。
- 新回归测试在修复前：1 failed，8 个应写入元素均不匹配。
- 修复后完整套件：25 passed，14 warnings。
- `git diff --check`：通过。

运行命令：

```bash
.venv/bin/python -m pytest tests/models/test_deepseek_v4_cpu_v028.py -q --disable-warnings
```

这些结果只证明已有局部测试及本次 cache 修复通过，不证明模型精度、
ARM Linux 端到端执行或性能达标。由于修复影响模型计算，正式 PR 前必须
运行真实模型评测。

## 补齐顺序与通过条件

1. 确认第一版权重格式。推荐已有 BF16 checkpoint；核对实际配置中是否
   仍含量化子模块，不能只看目录名。
2. 增加限定 DeepSeek V4 CPU 的显式 Torch 模式，配置检查应发生在加载
   权重之前；拒绝 strict、fused prepack 和暂未支持的量化组合。
3. 旁路 attention、mHC 和 linear 的 fused 准备，保留原始权重；在
   `fused_cpp` 已安装时也必须保持 Torch 行为，并记录实际后端。
4. 补充 BF16 Torch modular experts，复用已有 router；按 expert/token
   分块，避免展开为全部 token × top-k × 权重的巨大临时张量。
5. 覆盖 router FP32 语义、SwiGLU clamp、routed/shared scaling、hash
   路由、TP reduction、空 token 和 padding；通过模块级数值测试。
6. 在目标 ARM Linux 环境验证真实权重加载、prefill、decode 和连续生成，
   再运行模型评测。保存命令、模型 revision、代码版本和原始结果。

完成第 6 项前，不能将此分支描述为“纯 Torch ARM 已跑通”。
后续上游工作与 PR #55355 协调共同 CPU 模型和算子接口，避免重复提交
整套 CPU 模型骨架：<https://github.com/vllm-project/vllm/pull/55355。>
