# ARM CPU 实现与 PR #55355 的接口差异

比较日期：2026-09-07。

- 本地：`dsv4-arm-cpu-v0.28.0`，HEAD `31e4e4e3df`，包含当前未提交改动。
- PR：[#55355](https://github.com/vllm-project/vllm/pull/55355)，固定 head
  `7c10d5e680a4f69293306829f3777ea6e4599402`，仍为 Draft。
- 方法：检查本地当前源码、PR diff 以及该 head 的 attention 基类、CPU MoE
  完整源码。没有合并分支或执行两套后端的运行时互换测试。

本文的“不兼容”指直接替换会缺字段、使用错误布局或改变计算职责；
不意味着任一后端在自己的完整调用链中必然有错。

## 总体判断

外层模型/attention forward 和 modular experts 的接口已经接近，不需要
重新设计全部 ARM kernel。真正需要协调的是 cache 表示、metadata、indexer
返回值、shared expert 归属、mHC 融合边界及权重后处理顺序。

优先级：P0 为直接互换前必须解决；P1 为共享模型框架前需要明确；P2 为
能力范围差异，不必阻止第一版不使用该能力的贡献。

## 差异清单

### P0：主 attention / SWA / indexer cache 布局不同

本地 `cpu/attention.py`：

- SWA cache、主压缩 cache 和 indexer K cache 使用 BF16。
- cache 通过 `[block, offset, feature]` 寻址，支持 page padding。
- `get_kv_cache_spec()` 使用 BF16 dtype；本地 sparse backend 只声明
  `supported_kv_cache_dtypes = ["auto"]`。

PR `cpu/cpu_sparse.py::_fused_qnorm_rope_kv_insert()`：

- 明确断言 SWA cache 为 `torch.uint8`，只支持 `fp8_ds_mla`。
- 主 attention 的 packed cache 包含 FP8 数据、scale 与 BF16 RoPE 部分。
- indexer 基类使用 packed uint8 K cache；FP8 路径包含 128 个数据字节和
  4 个 FP32 scale 字节，而非 128 个 BF16 元素。
- `forward_mqa()` 将 paged cache 展平为 2D 并交给 C++ kernel。

后果：你的 cache 不能直接传给 PR kernel；其 uint8 cache 也不能直接由
你的 BF16 gather/attention 代码读取。修改 dtype 标记或函数名无法解决。
FP8 cache 还会改变数值路径，性能/精度对照必须记录 cache 格式。

建议：共同接口显式表达 cache layout/capability，保留独立 backend 和
KVCacheSpec。第一版 ARM 不必为了共用模型而强制改为 FP8 cache，也不应
每 token 转换整个 cache 来假装兼容。

### P0：metadata 类型与字段契约不同

本地：`cpu/sparse_mla.py` 定义自己的
`DeepseekV4CPUSparseMLAMetadata`、`DeepseekV4CPUSWAMetadata` 和
`DeepseekV4CPUCompressorMetadata`。主 sparse metadata 明确携带
`req_id_per_token`、`storage_block_size`、C4 prefill ranges、C128 top-k。

PR：`cpu/cpu_mla.py` 继承共享 sparse MLA/indexer/SWA builders，使用
`DeepseekV4FlashMLAMetadata`、`DeepseekSparseSWAMetadata`；compressor
还会断言 `isinstance(state_metadata, CompressorMetadata)`。
其 attention 从 SWA metadata 读取 `token_to_req_indices`，压缩 cache
寻址使用 `flashmla_metadata.block_size // compress_ratio`。

PR CPU builder 刻意不生成 C128 top-k metadata，由 `forward_mqa()` 根据
positions 计算；本地 fused sparse 路径使用本地 metadata 提供的数据。

后果：即使字段形状接近，也不能交叉使用 builder 和 consumer；可能直接
缺字段/类型断言失败，或因 logical block 与 storage block 混用而读错 cache。
两种 block-size 表达方式未必数值不同，但适配时必须验证单位关系。

建议：先对齐逻辑 token/物理 slot/压缩 block 的定义，再适配各自 builder；
不要通过去掉 assert 或简单字段重命名绕过。

### P0：indexer 的返回值和计算职责不同

本地 `DeepseekV4CPUIndexer.forward(...) -> Tensor`：
完成 compressor、query 投影、打分和 top-k，直接返回 `[T, topk]` 的
请求内压缩索引。attention 把该结果传给后续候选收集/稀疏计算。

PR 同名类的 `forward(..., qr_scale=None) -> (q, q_scale, weights)`：
一般返回 query 和 weights，后续 `indexer_op` 才写共享
`topk_indices_buffer`；短上下文分支直接填 buffer 并返回三个 None。

后果：替换类会让调用方将索引 Tensor 当成 tuple，或将 query tuple 当成
top-k；即使改了签名，也会漏算/重复算 indexer。

建议：明确公共边界选在“query preparation”还是“完整 indexer”；若保留
ARM 完整 indexer，需要在 adapter 中写公共 top-k buffer，并跳过重复打分。

### P0：shared expert 归属与权重所有权不同

本地：

- `FusedMoEFactory(shared_experts=None)`。
- 将 MLP 放到 `routed_experts._dsv4_shared_experts`，供 ARM 后端联合准备。
- `FusedCppArmExperts.fuses_shared_experts = True`，输出已包含 routed+shared。
- 后处理会释放 routed 权重及 shared MLP 原始权重。
- 未融合时由模型自行加 shared MLP，其 down projection 使用默认 reduction。

PR：

- 当前 `is_fused_shared_expert_enabled = False`。
- `shared_experts=self.shared_experts` 传给 factory/runner。
- shared MLP 的 `reduce_results=False`，计算组合和 reduction 交给既有
  MoE 执行链，不由模型 forward 再加一次。

后果：把 ARM backend 直接放进 PR 模型，会缺少 `_dsv4_shared_experts`
而失败；若只补属性，runner 仍可能重复执行 shared expert，或读取已被
ARM prepare 释放的空权重。还需确认 TP reduction 不重复、不遗漏。

建议：将“后端融合 shared expert”的能力与权重所有权显式传给 factory/
runner；或者首版 ARM 只处理 routed experts，随后独立接回联合融合。
保留融合时需要共同接口，不能只靠私有属性旁路。

### P0：全零 hash table 回退语义与 PR router 冲突

本地 `disable_placeholder_hash_routing()` 将全零表解释为 placeholder，
清除 router 的 `_hash_indices_table`，改走 score routing。
hash 层本身没有 `e_score_correction_bias`。

PR `CPURouter._compute_routing()` 对 sqrt-softplus 分支断言 hash table
或 correction bias 至少有一个；不提供两者皆 None 的分支，并使用
`hash_topk_cpu` / `biased_topk_cpu` 专用算子。

后果：将本地全零表回退原样带到 PR router，会触发断言。直接沿用 PR
行为则继续使用全零表，改变本地 checkpoint 的 expert 选择。

建议：先核实实际 checkpoint 中全零表是否确为 placeholder，再明确模型
策略及无 bias score-routing 接口。不能把这一模型语义差异当成普通重命名。

两边 GateLinear 都声明 FP32 输出，但本地额外强制/请求 fused_cpp 的
BF16 GEMM→FP32 路径。相同输出 dtype 不证明计算与舍入等价，需要比较
router logits 和实际 top-k。

### P1：mHC 融合边界、参数和 decoder 输入不同

本地：`cpu/mhc.py` 接受 prepared weight，融合 mHC pre/post-pre 与紧接的
RMSNorm；末尾还融合 post+head+RMSNorm。decoder 的 `_pre()` 在需要时
扩展残差流，第一层可从 `[T,H]` 输入开始。

PR：通过 `MHCPreOp`、`MHCPostOp`、`HCHeadOp` 分发；CPU wrapper 使用
原始权重。decoder 在 mHC 之后单独调用 `attn_norm` / `ffn_norm`，head
之后单独调用 `self.norm`。模型入口先将 embedding 扩展为 `[T,hc_mult,H]`。

具体参数差异：本地参考函数使用一个 `hc_eps`；PR wrapper 暴露
`hc_pre_eps` 和 `hc_sinkhorn_eps`，当前 CPU 实现要求二者相等，并要求
post multiplier 为 2.0。完整 `mhc_pre` 返回顺序都是
`(post_mix, comb_mix, layer_input)`，这部分无需重排；PR decoder 的
`hc_pre` helper 另行重排成 `(layer_input, post_mix, res_mix)`。

后果：直接将已融合 RMSNorm 的 ARM 函数注册为普通 MHCPreOp，会被模型
再次归一化；只替换第一层 decoder 还可能传错残差维度。prepared weight
也不能当作原始 Tensor 传给现有 wrapper。

建议：保留普通 mHC 算子契约；要保留 ARM 跨算子融合，则增加明确的
fused mHC+norm 路径及 capability，让调用方跳过对应 norm。

### P0：权重后处理顺序和布局不同

本地 attention 将多个 projection 标记为 `_cpu_keep_raw_weight` /
`_cpu_fused_cpp_joint_int8_owned`；deferred/model post-load 联合准备输入、
post、WO_A 等权重，strict 时才释放不再需要的原始张量。

PR 在 attention 构造时包装 `wo_a.quant_method.process_weights_after_loading`：
先反量化为 BF16、调用 `convert_weight_packed`，保存 `_wo_a_packed`，随后
执行原 hook 并释放原 weight/scale。输出阶段由 `bmm_cpu` 消费该 packed
对象。本地则使用 fused inverse-RoPE+WO_A 对象或原始 WO_A。

后果：PR 的 hook 若先执行，ARM 联合 prepare 可能只能看到空权重；若直接
复用 `_wo_a_packed`，其布局并不满足 ARM prepare 的原始 `[N,K]` 契约。

本地另有 `cpu_fp8_to_int8` 流式转换、TP 预切片标记以及
`VLLM_CPU_MOE_PREPACKED_DIR` 触发的 expert 权重跳读；这些不是 PR 原有
加载接口的一部分，必须单独迁移或禁用，不可默认兼容。

建议：每个 projection 的 prepare/release 只能由一个后端拥有；在原权重
被消费/释放之前完成选择，将转换及磁盘 prepack 作为独立能力验证。

### P1：torch.compile 与有状态 attention 的边界不同

本地 attention.forward 直接调用 Python `_forward_impl`，后者读写
forward context 中的 cache，并有依赖 metadata 的分支。

PR 使用注册的 `torch.ops.vllm.deepseek_v4_cpu_prepare_and_attn` opaque
边界，把 cache mutation 和输出投影一并包住。其注释明确说明：避免 CPU
`DYNAMO_TRACE_ONCE` 在 warmup metadata 为 None 时固化分支、消除真实计算。
PR 模型也有 `support_torch_compile` 装饰。

后果：eager 下能运行的 ARM Python 路径，不保证放进 PR 编译模型后仍正确。

建议：第一版明确使用经验证的 eager 设置；若接入 PR 编译路径，保留
有状态算子的边界并检查 fake implementation、输出形状与 warmup 后行为。

### P2：MTP 能力差异

本地 `cpu/mtp.py` 有完整的 predictor/权重加载实现，并有 MTP 专用权重
处理。PR 的同名入口 `DeepSeekV4MTP` 是直接抛出 NotImplementedError 的
占位类；它不提供可直接替换的 CPU MTP 实现。

建议：保留独立能力/后续 PR。不要用 PR 的文件覆盖本地后仍声称支持 MTP。

## 已接近兼容的部分

- `DeepseekV4ForCausalLM.forward(input_ids, positions, intermediate_tensors,
  inputs_embeds)`、`compute_logits`、`load_weights` 等外层入口相近。
- attention 外层 `forward(positions, hidden_states, llama_4_scaling=None)`
  相近；差异主要在内部状态和后端依赖。
- 本地 `FusedCppArmExperts` 与 PR `CPUUnquantizedExperts` 都使用 modular
  experts：`apply(output, hidden_states, w1, w2, topk_weights, topk_ids, ...)`
  的参数顺序相同，并使用 `TopKWeightAndReduceNoOP`。因此 routed experts
  是较好的适配切入点；shared 融合与权重生命周期仍须处理。
- BF16 MoE 输入、FP32 router logits 的目标一致，但不是数值等价证明。

## 建议讨论顺序

1. CPU cache layout 是否允许按后端选择 BF16 与 packed FP8；builder 和
   consumer 如何绑定，避免强制统一内部表示。
2. modular MoE 如何声明 routed+shared 融合，明确输出和 TP reduction 归属。
3. mHC 是否允许跨 norm 融合，以及原始/prepared 权重的生命周期。
4. 确定 indexer 公共边界后再适配 metadata；将 hash placeholder 策略作为
   明确的 checkpoint 兼容问题讨论。
5. 在共同模型上先验证 BF16 eager 的真实 prefill/decode；compile、量化、
   prepack 和 MTP 分别验收。

不建议整文件替换 `cpu/model.py` 后修到能 import 就认定兼容；这里多个
问题即使 Python 签名能对上，仍会改变实际计算。

## 源码定位

本地：

- [attention/cache/indexer](vllm/models/deepseek_v4/cpu/attention.py)
- [metadata/builders](vllm/models/deepseek_v4/cpu/sparse_mla.py)
- [model/MoE/decoder](vllm/models/deepseek_v4/cpu/model.py)
- [mHC adapters](vllm/models/deepseek_v4/cpu/mhc.py)
- [ARM modular experts](vllm/model_executor/layers/fused_moe/experts/fused_cpp_cpu_moe.py)

PR 固定版本：

- [attention/indexer](https://github.com/vllm-project/vllm/blob/7c10d5e680a4f69293306829f3777ea6e4599402/vllm/models/deepseek_v4/cpu/cpu_sparse.py)
- [metadata builders](https://github.com/vllm-project/vllm/blob/7c10d5e680a4f69293306829f3777ea6e4599402/vllm/models/deepseek_v4/cpu/cpu_mla.py)
- [compressor](https://github.com/vllm-project/vllm/blob/7c10d5e680a4f69293306829f3777ea6e4599402/vllm/models/deepseek_v4/cpu/cpu_compressor.py)
- [model](https://github.com/vllm-project/vllm/blob/7c10d5e680a4f69293306829f3777ea6e4599402/vllm/models/deepseek_v4/cpu/model.py)
- [CPU router](https://github.com/vllm-project/vllm/blob/7c10d5e680a4f69293306829f3777ea6e4599402/vllm/model_executor/layers/fused_moe/router/cpu_router.py)
- [CPU mHC](https://github.com/vllm-project/vllm/blob/7c10d5e680a4f69293306829f3777ea6e4599402/vllm/model_executor/kernels/mhc/cpu.py)
- [CPU MTP stub](https://github.com/vllm-project/vllm/blob/7c10d5e680a4f69293306829f3777ea6e4599402/vllm/models/deepseek_v4/cpu/mtp.py)
