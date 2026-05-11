from vllm import LLM, SamplingParams
import time
import os

# 🔪 强行关闭 V1 引擎，使用经典引擎，逼出真实报错日志！
os.environ["VLLM_USE_V1"] = "0"
# 开启所有 debug 日志
os.environ["VLLM_LOGGING_LEVEL"] = "INFO"
def main():
    # ⚠️ 替换为你本地已经下载好的模型路径。
    # 建议先用一个小模型（比如 facebook/opt-125m 或 Qwen-1.8B）跑通流程
    MODEL_PATH = "/mnt/models/DeepSeek-R1" 

    print("🚀 正在初始化 vLLM 引擎...")
    # 🔪 核心触发器：tensor_parallel_size 必须设为 4，这会强行激活 AllReduce 逻辑！
    llm = LLM(
        model=MODEL_PATH, 
        tensor_parallel_size=4, 
        trust_remote_code=True,
        enforce_eager=True  # CPU 下开启 eager 模式能避开计算图捕获的各种奇怪报错
    )

    prompts = ["The future of artificial intelligence in high performance computing is"]
    # 生成 50 个 token，确保有足够的 Decode 循环来放大通信开销
    sampling_params = SamplingParams(temperature=0.0, max_tokens=50)

    print("\n🔥 开始预热 (Warmup)...")
    llm.generate(prompts, sampling_params)

    print("\n⏱️ 开始正式性能测试...")
    start_time = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params)
    end_time = time.perf_counter()

    # 📊 提取并计算关键指标
    token_count = len(outputs[0].outputs[0].token_ids)
    total_time = end_time - start_time
    tpot = (total_time / token_count) * 1000  # 转换为毫秒

    print("\n" + "="*50)
    print(f"✅ 生成完毕！总 Token 数: {token_count}")
    print(f"📊 整体总耗时: {total_time:.3f} 秒")
    print(f"🚀 每 Token 生成延迟 (TPOT): {tpot:.2f} 毫秒/Token")
    print("="*50)

if __name__ == "__main__":
    main()