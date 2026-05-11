import torch
import os
import glob
import time
# 自动寻找并加载编译好的 .so 库
import vllm
vllm_path = os.path.dirname(vllm.__file__)
so_files = glob.glob(os.path.join(vllm_path, "*.so"))
for so in so_files:
    torch.ops.load_library(so)

vllm_ops = torch.ops.vllm

# 设置随机种子和数据规模
# 2048 是你刚才测试的长度，你可以尝试从小到大测试
def benchmark(numel):
    print(f"\n--- 测试规模: {numel} elements ---")
    
    # 准备数据 (必须是连续的，且在 CPU 上)
    dst = torch.randn(numel, dtype=torch.float16)
    src = torch.randn(numel, dtype=torch.float16)
    
    # 为了公平，先预热（Warmup）几次，让 CPU 进入高频状态
    for _ in range(10):
        vllm_ops.kunpeng_all_reduce(dst, src)
        dst += src

    # --- 测试 PyTorch 原生 (dst += src) ---
    iters = 10000
    start = time.perf_counter()
    for _ in range(iters):
        # 模拟 Reduce 里的加法操作
        dst.add_(src) 
    end = time.perf_counter()
    torch_time = (end - start) / iters * 1e6 # 转换成微秒 (us)
    print(f"PyTorch 原生加法平均耗时: {torch_time:.3f} us")

    # --- 测试你的 SVE 汇编算子 ---
    start = time.perf_counter()
    for _ in range(iters):
        vllm_ops.kunpeng_all_reduce(dst, src)
    end = time.perf_counter()
    sve_time = (end - start) / iters * 1e6
    print(f"鲲鹏 SVE 汇编算子平均耗时: {sve_time:.3f} us")

    speedup = torch_time / sve_time
    print(f"🚀 核心算子加速比: {speedup:.2f}x")

# 测试不同规模
for size in [1024, 2048, 4096, 8192, 16384, 131072]:
    benchmark(size)