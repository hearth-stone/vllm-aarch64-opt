<!-- Copyright (c) 2026 Huawei Technologies Co., Ltd. -->

# DeepSeek V4 CPU service and request tests

## Start

This launcher uses the existing Arm-codex-internal environment: TP4 with 80
CPUs/rank on NUMA nodes 0/1/2/3, Python
`/home/zhangxu/final_test/test/bin/python`, LLVM OpenMP, PyTorch's built-in mimalloc plus preloaded jemalloc, and the
SiLU/mHC-fixed fused_cpp in `/home/zhangxu/final_test/fused_cpp`. Run from
`/home/zhangxu/final_test/vllm-aarch64-opt`.

Select the model in `scripts/serve_deepseek_v4_cpu.sh` by leaving exactly one
of these assignments uncommented. vLLM reads quantization from `config.json`.

```bash
# model=/mnt/models/DeepSeek-V4-Flash-BF16
# model=/mnt/models/DeepSeek-V4-Flash-INT8
model=/mnt/models/DeepSeek-V4-Pro-Base
```

Then start it without a model argument:

```bash
bash scripts/serve_deepseek_v4_cpu.sh
```

Run one service at a time. Each command runs in the foreground and prints a new
log directory under `logs/MODEL-TIMESTAMP/`, containing `server.log`,
`server.pid` and `model-path.txt`. To launch in the background:

```bash
nohup bash scripts/serve_deepseek_v4_cpu.sh \
  > /tmp/deepseek-v4-launch.log 2>&1 &
```

The launcher activates the uv-created environment with
`source /home/zhangxu/final_test/test/bin/activate`, then starts vLLM using
`python`.

The defaults are maximum context/batched-token budget 9216 and KV cache
32 GiB/rank. Edit these assignments in the script to change them. The script
uses these values directly and does not accept positional arguments or external
overrides for them. For example:

```bash
export MAX_MODEL_LEN=4096
export KV_GIB_PER_RANK=32
```

The service is `http://127.0.0.1:8018`, with model ID `deepseek-v4`. Wait for
startup completion in `server.log`, then check
`curl http://127.0.0.1:8018/health`. Use `nice -n -20 bash ...` externally if
the account has the previously configured priority permission; the launcher
does not require that permission.

Use `/mnt/models/DeepSeek-V4-Pro-Base` directly for Pro. The launcher reads
`config.json`; for a DeepSeek V4 FP8 checkpoint it supplies `--hf-overrides`
that enable `cpu_fp8_to_int8` and the existing CPU W8A8 configuration. The
original model directory and configuration are unchanged. Overrides are saved
as `hf-overrides.json` in the run log directory.

During loading, `cpu/model.py` calls the streaming converter in
`cpu/fp8_requant.py`, producing INT8 weights/scales for W8A8 target layers and
BF16 for other layers, then the CPU operators pack the resulting weights.
Flash BF16 and INT8 receive an empty override and retain their own configuration.
No separate runtime view or offline checkpoint conversion is needed. Config
validation does not constitute a fresh full-model Pro serving validation.

## Runtime settings

The launcher exposes two settings and retains twelve runtime variables (the prepack path applies only to Flash BF16):

| Variable | Purpose |
| --- | --- |
| `MAX_MODEL_LEN` | Maximum context length and batch token budget |
| `KV_GIB_PER_RANK` | KV capacity in GiB per TP rank |
| `PYTHONPATH` | Load this checkout and fused_cpp |
| `LD_PRELOAD` | Use LLVM OpenMP and jemalloc for standard malloc/free |
| `MIMALLOC_PURGE_DELAY` | Disable purging in PyTorch's built-in mimalloc |
| `MALLOC_CONF` | Retain reusable allocations with the tested allocator policy |
| `VLLM_TARGET_DEVICE` | Select CPU |
| `VLLM_CPU_FUSED_CPP_STRICT` | Enable the deployed fused DeepSeek V4 CPU paths |
| `VLLM_CPU_OMP_THREADS_BIND` | Supply the explicit per-rank CPUs required by Plan V2 |
| `FUSED_CPP_MOE_PREPACK_THREADS` | Use 16 threads/rank for startup expert-weight packing |
| `GLOO_SOCKET_IFNAME` | Use the machine's `eno1` interface for TP communication |
| `FUSED_CPP_MOE_PLANNER_PROFILE` | Use the existing rank0-value profiles for all ranks |
| `VLLM_CPU_MOE_PREPACKED_DIR` | Load the existing Flash BF16 expert cache; unset for other models |
| `VLLM_CPU_MOE_PREPACK_PREFAULT` | Touch each cached weight page during startup, before serving |

NUMA placement and KV capacity are CLI arguments. There is no profiler,
long timeout override or priority preflight. The launcher uses the existing
fixed profiles `/home/zhangxu/final_test/cpu_runtime/profiles/rank{tp_rank}.json`,
whose ranks share rank0 calibration values. Flash BF16 also loads
`/mnt/models/.cache/zhangxu/DeepSeek-V4-Flash-BF16-moe-prepacked-v028-482e03e`.
This cache applies only to `/mnt/models/DeepSeek-V4-Flash-BF16`; INT8 and Pro
clear the prepack path and pack their weights at startup. The legacy KV
capacity override is cleared. Cached weights are prefaulted at startup by touching one byte per4096-byte
page and the tail; this shifts first-touch costs to startup and does not pin
the pages permanently. Requests reuse packed weights in memory.
These changes take effect on the next service start; earlier fresh-calibration
and no-cache results below retain their original experiment settings.

CPU tensors retain the original PyTorch allocator (built-in mimalloc v2.3.2
on this machine); standard malloc/free calls use preloaded jemalloc. Both
allocators retain reusable pages. The custom allocator op and bootstrap have
been removed, so `libvllm_cpu_allocator.so` is no longer required. Use a fresh
shell without previously imposed OpenMP/MKL thread limits when launching the
80-core configuration.

## Test

The client repeats the existing `zh2048-001` prompt: 2048 input tokens,
one output token, temperature zero and top-5 logprobs. The default is three
warmup requests followed by sixty measured requests with a five-second gap
after each response. Warmups are saved separately and excluded from summary.

```bash
/home/zhangxu/final_test/test/bin/python scripts/test_deepseek_v4_cpu_api.py \
  --model deepseek-v4 --service-run logs/DeepSeek-V4-Flash-BF16-YYYYMMDD-HHMMSS \
  --output-dir /home/zhangxu/final_test/test-bf16-gap5-RUN \
  --requests 60 --interval-seconds 5 --warmup 3

/home/zhangxu/final_test/test/bin/python scripts/test_deepseek_v4_cpu_api.py \
  --model deepseek-v4 --service-run logs/DeepSeek-V4-Flash-INT8-YYYYMMDD-HHMMSS \
  --output-dir /home/zhangxu/final_test/test-w8a8-continuous-RUN \
  --requests 60 --interval-seconds 0 --warmup 3
```

Use `--interval-seconds 0` for serial continuous requests: the next request
starts after the previous response. This does not create concurrent requests.
Replace `YYYYMMDD-HHMMSS` with the log directory printed at launch.
Use a new output directory for each test. The client checks the served model
ID and request token counts, and records per-rank faults and RSS. Outputs:
`protocol.json`, `warmup.json`, `requests.json`, and `summary.json`.

The launcher uses the served model ID `deepseek-v4` for every model.
These are serving/performance checks for one repeated input. They do not
replace model accuracy evaluation or a concurrency benchmark.

For five different Chinese inputs of a given length, pass the corresponding
dataset, `--input-tokens`, and `--distinct-cases`. The first five cases are
measured; the first case is also used for the two excluded warmups below.
The existing dataset directory contains 100 cases per length. Repeat with
`8192` in both the dataset path and token argument for the 8K test, or change
the model path and service directory for W8A8.

```bash
/home/zhangxu/final_test/test/bin/python scripts/test_deepseek_v4_cpu_api.py \
  --model deepseek-v4 --service-run logs/DeepSeek-V4-Flash-BF16-YYYYMMDD-HHMMSS \
  --output-dir /home/zhangxu/final_test/test-bf16-4096-RUN \
  --cases /home/zhangxu/final_test/datasets/chinese_prefill_20261010/cases_4096.jsonl \
  --input-tokens 4096 --distinct-cases \
  --requests 5 --interval-seconds 0 --warmup 2
```

## Continuous-request results, 2026-10-10

Each variant completed three warmups and sixty serial requests with no sleep.
The same input, runtime, calibration values and allocator policy were used.
W8A8 reused the already loaded service; BF16 was restarted with this launcher.

| Variant | Previous gap-5 mean, 60 requests | Continuous mean | Continuous P95 | Maximum |
| --- | ---: | ---: | ---: | ---: |
| BF16 | 4.226 s | 5.948 s | 6.383 s | 6.556 s |
| W8A8 | 2.781 s | 3.873 s | 4.254 s | 4.491 s |

Immediately subsequent three-request gap-5 controls, using the same client and
service, returned to 4.210 s for BF16 and 2.848 s for W8A8. The runs therefore
show a cadence-dependent latency increase; the mechanism has not been profiled.
Continuous runs had only 178/12 total minor faults across all ranks for
BF16/W8A8 and zero major faults. BF16 RSS rose less than 0.36 MiB per rank;
W8A8 RSS did not grow. All sixty outputs and top-5 logprobs matched the previous
same-variant reference, with maximum absolute logprob difference zero.

Artifacts are under
`/home/zhangxu/final_test/tp4_diagnose_20261010/continuous60-{bf16,w8a8}-20261010`
and corresponding `continuous60-{bf16,w8a8}-gap5-control-20261010` directories.
The aggregate record is `continuous60-comparison-20261010.json` in their parent.

## 4K/8K results, 2026-10-10

Each variant used maximum model length 8256 and 8 GiB KV cache, followed by
two warmups and five different inputs per length. Requests were serial with
no sleep, generated one token, and returned the exact requested input lengths.
The first measured case was also used for warmup. Both variants used the same
five cases at each length, with the runtime and calibration described above.

| Input tokens | Variant | Mean | Median | Minimum | Maximum |
| ---: | --- | ---: | ---: | ---: | ---: |
| 4096 | BF16 | 12.259 s | 13.020 s | 9.356 s | 13.499 s |
| 4096 | W8A8 | 9.819 s | 10.459 s | 6.607 s | 10.988 s |
| 8192 | BF16 | 26.219 s | 26.509 s | 24.921 s | 26.711 s |
| 8192 | W8A8 | 20.419 s | 21.678 s | 15.676 s | 22.534 s |

These five-request groups include different news texts and should not be
treated as a direct length-scaling comparison against the historical repeated
2048-token prompt. Minor faults were substantial on the longer, distinct
inputs; all four groups had zero major faults during measurement. No profiler
recording was enabled and fault cost was not attributed in this run.

Artifacts are under
`/home/zhangxu/final_test/tp4_diagnose_20261010/length5-{bf16,w8a8}-{4096,8192}-20261010`.
The aggregate record is `length5-4k8k-comparison-20261010.json` in their parent.

## Complete-answer batch acceptance

`accept_deepseek_v4_cpu.py` prepares and runs a separate complete-answer test.
It uses the deployed DeepSeek V4 chat encoding in non-thinking mode, sends
pre-tokenized prompts through streaming completions, and permits normal EOS.
It does not force a JSON schema during generation.

The first comparison uses 24 controlled Chinese extraction cases: 4/4/8/4/4
cases at exactly 512/1024/2048/4096/8192 input tokens including chat framing.
Each contains one explicitly fictional test record with four gold string
fields (identifier, institution, date, count), inserted near the beginning,
middle or end of existing licensed news text. This tests extraction and
cross-request contamination; it is not a general news QA accuracy evaluation.
Prompts, expected fields and news source attribution are saved with the cases.

```bash
/home/zhangxu/final_test/test/bin/python scripts/accept_deepseek_v4_cpu.py prepare \
  --dataset /home/zhangxu/final_test/datasets/chinese_prefill_20261010 \
  --tokenizer /mnt/models/DeepSeek-V4-Flash-BF16/tokenizer.json \
  --output /home/zhangxu/final_test/tp4_acceptance_20261010/cases24.jsonl

/home/zhangxu/final_test/test/bin/python scripts/accept_deepseek_v4_cpu.py run \
  --model deepseek-v4 \
  --cases /home/zhangxu/final_test/tp4_acceptance_20261010/cases24.jsonl \
  --service-run logs/DeepSeek-V4-Flash-BF16-YYYYMMDD-HHMMSS \
  --output-dir /home/zhangxu/final_test/accept-bf16-RUN \
  --batch-sizes 1 2 4 8 --max-tokens 256 --warmup 3 \
  --stress-seconds 600 --stress-concurrency 8
```

For INT8 use the W8A8 model directory and the same service ID `deepseek-v4`. Each batch wave
starts its requests together and waits for all responses. Sustained pressure
keeps eight requests in flight for at least 600 seconds, then drains them.
The three warmups are excluded from measured groups. Requests are not retried.

An additional native API batch check sends multiple prompts in one HTTP
request. It reuses the first eight cases (four 512-token and four 1024-token
inputs) at B=2/4/8 and verifies choice indices, gold fields and agreement with
single-request answers. Its nonstream latency covers the complete HTTP batch;
TTFT/TPOT measurements come from the main streaming experiment above.

```bash
/home/zhangxu/final_test/test/bin/python scripts/accept_deepseek_v4_cpu.py api-batch \
  --model deepseek-v4 \
  --cases /home/zhangxu/final_test/tp4_acceptance_20261010/cases24.jsonl \
  --baseline /home/zhangxu/final_test/tp4_acceptance_20261010/bf16-initial/batch1-requests.json \
  --output-dir /home/zhangxu/final_test/accept-bf16-native-RUN --limit 8
```

A correct answer must parse as the gold four-field JSON object and finish
normally; optional JSON code fences are accepted. Wrong/missing/extra fields,
empty outputs, truncation, replacement characters and invalid control
characters are reported. Every request retains its complete text, SSE events,
token usage, finish reason, errors, E2E latency, streaming TTFT and mean TPOT.
Group summaries include P50/P90/P95/P99, output throughput and semantic
agreement with the same variant's single-request answers. HTTP metrics sampled
every 0.2 seconds record observed running requests and worker survival; requested
client concurrency must be reported separately from observed server load.

The occupation file for this experiment is
`/home/zhangxu/arm-codex-experiment-status.json`. It contains only `status`,
`phase`, and `results_dir`. The experiment controller updates progress and sets
`status` to `released` only after both variants finish and their services stop.
If a test fails, it leaves the marker occupied with an inspection message.

```bash
cat /home/zhangxu/arm-codex-experiment-status.json
```

The initial INT8 full-answer probe on 2026-10-10 failed on its first decode
step with an unprepared joint INT8 projection. The machine still had the old
`CPUInt8ScaledMMLinearKernel` error path; the local implementation already
provided unpacked per-token W8A8 projection for decode. That file was synced
after preserving the deployed original and crash log. Independent integer
accumulation checks passed at shapes `(1,32)`, `(2,32)`, `(8,32)`, `(2,4,32)`,
including zero input and bias. Full-model INT8 acceptance must use and report
this updated deployment; the failed original probe remains part of the record.

The original BF16 run completed 24/24 correct single answers, then failed in
the second B=2 wave when one request decoded while another prefetched. Strict
SWA metadata omitted prefill index rows although mixed-batch attention consumed
them. `DeepseekV4CPUSWAMetadataBuilder` now retains those rows for mixed batches
and still skips them for pure fused prefill. Five focused Arm regressions passed,
including strict/non-strict mixed metadata, invalid tokens, cache row selection
and the existing pure-prefill allocation check. The original results remain in
`bf16-initial`; fresh complete runs use `bf16-mixed-fixed` and `int8-mixed-fixed`.

The fixed BF16 concurrent B=2 run completed 20 correct answers, then two 8K
requests timed out at 900.10/926.87 seconds. One returned only `{"`; neither
is a correct complete answer. The client and controller terminated; workers
were still alive before intentional experiment cleanup. Preserve these as
request timeouts, separately from spontaneous engine crashes.

To continue independent phases, `run --baseline PATH` reuses the same model's
saved single-request answers. It checks case IDs, lengths and gold fields.
For example, pass `--batch-sizes 4 --stress-seconds 0` to run only B=4, or
an empty `--batch-sizes` with `--stress-seconds 600` for pressure only. Use a
new output directory. A failed wave is saved with incomplete coverage marked;
already started pressure requests are drained and all outcomes retained.
No failed request is retried or omitted from its original phase.

The experiment continuation controller records independent terminal phases in
`continuation-phase-results.json`. It first completes native API comparison and
INT8 phases, then attempts the remaining BF16 cases and B=4/8/pressure phases.
Later phases can run after a recorded failure and intentional service cleanup.
The occupation marker remains occupied until all phases terminate and all
experiment services stop; released means the experiment ended, not that it
passed acceptance.

### Candidate memory in mixed prefill/decode

The INT8 B=2 continuation with 8192-token inputs and 128 GiB KV per rank
failed with two `EngineDeadError` streams. Worker TP3 exited with code -9;
the account cgroup recorded one OOM kill and a peak of 2,125,294,374,912 bytes.
The kernel's killed-PID record requires sudo, so OOM is the probable cause
rather than a confirmed kernel attribution. The failure is retained in
`tp4_acceptance_20261010/int8-8k-engine-failure.json`.

The mixed attention path previously retained a candidate-KV tensor for every
query and concatenated them into another tensor. It now gathers and executes
at most 32 queries at a time, retaining global token indices and the original
scale, sink, and output ordering. Pure fused prefill keeps its existing path.
The old gather list is released before building the next chunk.

The Arm regression command selected `sparse_candidate_chunks`,
`mixed_swa_metadata`, and `strict_swa_metadata` in
`tests/models/test_deepseek_v4_cpu_v028.py`: 9 tests passed. These check FP32 and
BF16 outputs against a float64 attention oracle, fused/reference dispatch,
query ordering across chunk boundaries, and the prior mixed metadata fix.
Full-model revalidation of the bounded candidate path is pending. New results
must identify this source change and use a fresh single-request baseline at
128 GiB KV per rank before the complete B=2/4/8 and sustained-load matrix.

## Launcher validation, 2026-10-11

The generic launcher passes `bash -n` and ShellCheck. Its request client passes
Ruff and CLI help checks. On Arm-codex-internal, the actual vLLM CLI imports
successfully without the removed `LD_LIBRARY_PATH` override. Creating engine
configurations for all three model directories identifies BF16 as unquantized
and Flash INT8 / Pro runtime as `compressed-tensors`, with BF16 activations,
TP4 and the requested 32 GiB KV/rank. These checks do not load checkpoint
weights or constitute fresh model serving/performance tests.

## Original Pro FP8 serving and performance, 2026-10-11

The generic launcher was tested with `/mnt/models/DeepSeek-V4-Pro-Base` directly,
using its automatic HF overrides for streaming CPU conversion. The first launch
completed FP8 loading/conversion, but startup perf showed serial expert packing
(default one thread/rank) dominated preparation. That run was intentionally
stopped before inference. The launcher now sets
`FUSED_CPP_MOE_PREPACK_THREADS=16`; the corresponding W8A8 MoE tests passed
9/9. This controls startup packing, while inference keeps 80 CPUs/rank.

The updated launcher completed serving successfully in 1341.53 seconds
(22.36 minutes) from launch to health. It uses TP4, LLVM OpenMP, jemalloc,
NI=-20 supplied externally, context/batch budget 9216 and KV 32 GiB/rank.
The checkpoint is approximately 1495.75 GiB. Each rank logged conversion of
71764 matrices to INT8 and 188 to BF16; those are loader-stream counts.
No original checkpoint/configuration was modified, and no disk prepack cache
or fixed calibration table was used.

The performance client used the historical `zh2048-001` input: 2048 input
tokens, one output token, temperature zero, top-5 logprobs, three excluded
warmups and ten measured requests, with ten seconds after each response.

| Metric | Result |
| --- | ---: |
| Mean | 9.051265 s |
| Median | 9.030475 s |
| P95 | 9.466022 s |
| Minimum | 8.600795 s |
| Maximum | 9.538945 s |
| Input throughput, 2048 / mean | 226.267 tokens/s |

All ten requests succeeded and returned the same text (`流动`) and top-5
logprobs. No timeout, engine crash or OOM occurred. This is a repeated-input,
single-token performance check rather than general accuracy evaluation.
The mean is numerically 8.96% above the historical Pro W8A8 mean 8.307201 s
(N=60, gap10). Sample count, fresh rank-local calibration and runtime settings
are different, so this is not a controlled regression comparison.

Artifacts and complete responses are in
`/home/zhangxu/final_test/generic-v4-pro-perf-20261011-prepack16`.
The serial-packing diagnosis remains in
`/home/zhangxu/final_test/generic-v4-pro-perf-20261011`.
All recorded owned service PIDs were absent after cleanup, port 8018 was closed
and the occupation marker was released. Pro is now the uncommented model in
the launcher; starting it again performs another cold start.

## Default hybrid allocator, 2026-10-11

The default is now PyTorch mimalloc (`MIMALLOC_PURGE_DELAY=-1`) plus jemalloc
for standard malloc/free, with the retention settings above. No custom C10
allocator is installed. The BF16 and Flash INT8 service checks each used
TP4x80, NI=-20, the same rank0-value MoE profiles, context/batch budget9216,
KV32GiB/rank, 2048 input tokens, one output token, three excluded warmups and
ten measurements with five seconds after each response. BF16 used its disk
prepack cache; INT8 packed at startup. These controlled experiment settings
differ from the generic launcher's fresh calibration and no disk cache.

| Model | Mean | P95 | Minor faults/rank, total over ten requests |
| --- | ---: | ---: | ---: |
| Flash BF16 | 4.095657 s | 4.128694 s | 20–22 |
| Flash INT8 / W8A8 | 2.751914 s | 2.781962 s | 50–96 |

All requests succeeded, with zero major faults. For each model, all 39
responses across the earlier jemalloc, mimalloc and hybrid checks (including
warmups) had identical output tokens and top-five logprobs. These are fixed
input, single-token checks, not full accuracy or long-running batch validation.
Pro's earlier results above used the previous custom tensor allocator.
The hybrid-default Pro run is recorded below. Detailed Flash records are in
`/home/zhangxu/final_test/hybrid-allocator-20261011`.

## Pro hybrid default validation, 2026-10-11

The actual default launcher completed serving the original
`/mnt/models/DeepSeek-V4-Pro-Base` FP8 checkpoint with online CPU W8A8
conversion, PyTorch's original mimalloc plus preloaded jemalloc. All four
ranks were verified to use the default tensor allocator, load jemalloc, set
`MIMALLOC_PURGE_DELAY=-1`, and omit the independent allocator library.
TP4x80, NI=-20, context/batch budget9216, KV32GiB/rank, packing16 threads,
fresh rank-local calibration and no disk prepack cache matched the earlier
generic Pro run.

Startup to health took 2185.46 seconds (36.42 minutes), versus 1341.53 seconds
in the previous run. Rank3 finished conversion/calibration later than
rank0–2. A separate three-second rank3 startup perf sample was dominated by
LLVM OpenMP wait/barrier symbols, with PyTorch multiply/copy/round/reduction
operations also present. This sample is not a complete startup profile and
does not establish an allocator cause; perf reported one lost event chunk
and zero lost samples. No profiler ran during measured inference.

The request protocol was identical to the prior Pro check: `zh2048-001`,
2048 input tokens, one output token, temperature0, top5logprobs, three
excluded warmups and ten measurements, ten seconds after each response.

| Metric | Hybrid default | Previous custom tensor jemalloc |
| --- | ---: | ---: |
| Mean | 8.954076 s | 9.051265 s |
| Median | 8.877642 s | 9.030475 s |
| P95 | 9.319079 s | 9.466022 s |
| Minimum | 8.702340 s | 8.600795 s |
| Maximum | 9.434433 s | 9.538945 s |

Mean latency is numerically 1.07% lower; fresh calibration and different run
times prevent attributing this small difference to the allocator. All ten
requests succeeded. All thirteen responses including warmups had identical
output text and top5logprobs within this run and against the prior Pro
reference. Minor faults over ten requests, ranks0/1/2/3:56/101/101/62;
major faults:1/0/0/0. No request timeout, model crash or OOM occurred.
Sampled account peak memory including file cache was1.9231TiB.

Complete results, deployed launcher/worker copies, four calibration tables,
startup perf and service cleanup records are in
`/home/zhangxu/final_test/pro-hybrid-allocator-20261011`. All owned model
service PIDs were absent after cleanup and port8018 was closed. This remains
a fixed-input single-token check, not full model accuracy or multi-batch
acceptance.

## final_test fused_cpp deployment, 2026-10-11

The default `fused_root` is `/home/zhangxu/final_test/fused_cpp`, updated to
delivery commit `c095846` on `feat/fused_cpp/dsv4_v028_aot_delivery`. This includes
the SiLU coefficient fix (`744855b`) and continuous-K mHC fix (`c095846`).
The existing local planner direct-route offset guard is preserved; it is the
only remaining source modification in that remote checkout.

Rebuilt in place with GCC13.2, Python3.12/PyTorch2.13 CPU, SVE256 and MAX_JOBS16.
Focused mHC/SiLU/W8A8 tests:110 passed. Complete delivery suite:1633 passed,
1 skipped in48.89s. Both native modules were verified to load from this
directory, and vLLM CLI import was checked with the updated path. Earlier model
serving results used the equivalent candidate kernels; full model serving was
not repeated for this path-only deployment change. Build/test logs and the
previous source/binary backup are in
`/home/zhangxu/final_test/final-test-fused-update-20261011`.
