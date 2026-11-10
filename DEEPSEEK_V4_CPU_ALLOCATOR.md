# DeepSeek V4 CPU allocator experiment, 2026-10-10

> Historical experiment: superseded on 2026-10-11 by the hybrid default.
> PyTorch retains its built-in mimalloc with `MIMALLOC_PURGE_DELAY=-1`;
> preloaded jemalloc handles standard malloc/free. The custom allocator source,
> binding, build integration and bootstrap were removed. Current settings and
> validation are in [the service guide](scripts/DEEPSEEK_V4_CPU.md#default-hybrid-allocator-2026-10-11).
> The former integration is archived outside the checkout in the experiment
> records; the descriptions and measurements below are historical.

The Arm-codex-internal PyTorch 2.13 CPU build calls its bundled mimalloc from
`c10::alloc_cpu`. Preloading jemalloc therefore did not change the allocator
used by ordinary CPU tensors. A real service fault profile included
`libc10.so::mi_segment_alloc` and MoE W2 stores.

`csrc/cpu/allocator.cpp` installs a C10 CPU allocator backed by the already
loaded jemalloc DSO. Allocation and release use callbacks from the same DSO.
The allocator preserves PyTorch alignment, NUMA movement, fill flags and CPU
memory-profiler reporting. Existing tensors retain their original deleters.
CPUWorker installs it after NUMA setup and before model loading. Environments
without preloaded jemalloc retain their default allocator.

## Matched service results

Machine: Arm-codex-internal, TP4, 80 CPUs per rank, NUMA nodes 0/1/2/3, LLVM
OpenMP, NI=-20. Same rank0-value calibration profiles, prepacked MoE cache and
BF16 clamp-fix kernels throughout. Requests use `zh2048-001`: 2048 input tokens,
one output token, temperature 0, top-5 logprobs, prefix caching disabled.
Each phase has one warmup, ten measured requests at five-second intervals,
and one separate Torch-profiled request. Profiled requests are excluded from
the latency statistics.

| Allocator and configuration | Mean request | Minor faults/request/rank |
| --- | ---: | ---: |
| Original mimalloc tensors; jemalloc preloaded | 4.8621 s | 245,323 |
| Actual jemalloc; original 30-second decay | 6.1945 s | 1,146,877 |
| Actual jemalloc; oversize disabled, 30-second decay | 6.5513 s | 1,340,441 |
| Actual jemalloc; oversize disabled, decay disabled | 4.2180 s | 297 |

The final configuration reduces mean request latency by 13.25%. Its measured
range is 4.1820–4.2508 s. The first measured request still has approximately
2,940 minor faults per rank; the following nine have 0–11. All measured phases
have zero major faults. Startup/warmup faults are retained in the raw records
but excluded from the measured-request statistics.

The active launch uses:

```text
LD_PRELOAD=/usr/local/lib/libomp.so:/usr/lib64/libjemalloc.so
MALLOC_CONF=background_thread:true,metadata_thp:auto,oversize_threshold:0,dirty_decay_ms:-1,muzzy_decay_ms:-1
```

The original oversize threshold is 8 MiB. A 224 MiB allocation was observed in
arena 1280 with dirty/muzzy decay of zero despite the global 30-second setting.
Disabling oversize alone did not resolve the service fault rate; disabling
decay was also necessary in this experiment. The final benefit combines
allocator registration and retention configuration.

The service routes six experts plus one shared expert. Its internal FP32
route buffer is 2048 × 7 × 4096 × 4 = 224 MiB per layer/rank. Passing vLLM's
reusable output tensor does not remove this internal allocation.

## Torch profile and correctness

The following totals are averages across four ranks, from one matched CPU
Torch profile per phase. They are not unprofiled latency measurements.

| Scope | Original | Final |
| --- | ---: | ---: |
| MoE, 43 calls/rank | 2535.06 ms | 2114.80 ms |
| shm_allreduce, 87 calls/rank | 750.13 ms | 936.16 ms |

MoE scope time decreases by 16.58%. All-reduce does not decrease in this
profile: final per-rank totals are 1239.04/163.27/1194.71/1147.61 ms. These
durations include rank-arrival waiting and overlap across ranks. This
experiment does not establish that communication imbalance is resolved.

All twelve responses in every phase match the baseline token and top-5
logprobs; maximum absolute logprob difference is zero. The allocator regression
test passes both default and jemalloc cases, including release of tensors
created before switching, alignment, cloning, actual jemalloc arena membership,
and CPU memory-profiler reporting. Compiler warnings are errors; Ruff,
clang-format and diff checks pass.

At the last measured request, mean RSS per rank is approximately 147.90 GiB
for the baseline and 147.19 GiB for the final configuration. Disabling decay
retains the allocation high-water mark; this single repeated input does not
establish memory usage for other input lengths or concurrency.

## Deployment and artifacts

The source change is integrated into CPU extension CMake sources and Torch
bindings. On the machine, the allocator source was separately compiled into
`libvllm_cpu_allocator.so` and registered with the same `_C::init_cpu_allocator`
schema. A task-local `sitecustomize.py` on PYTHONPATH loads that DSO alongside
the existing `_C.abi3.so`. The full vLLM `_C` extension was not rebuilt; the
native allocator and its worker integration were tested in the live service.

Remote experiment directory:
`/home/zhangxu/final_test/tp4_diagnose_20261010/allocator-fix-20261010`.
It contains the native source/binding, regression test, request client, four
phases of requests and profiles, and `comparison.json`. The local mirror is
`../tmp/tp4_diagnose_20261010/allocator-fix-20261010`.

Active service directory:
`/home/zhangxu/final_test/tp4_diagnose_20261010/serve-jemalloc-nodecay-ni20-20261010`.
Its launch script records the allocator bootstrap and retention settings.

Validation command (using the existing machine environment):

```bash
VLLM_ALLOCATOR_TEST_LIBRARY=/home/zhangxu/final_test/tp4_diagnose_20261010/allocator-fix-20261010/libvllm_cpu_allocator.so \
  /home/zhangxu/final_test/test/bin/python -m pytest -q \
  /home/zhangxu/final_test/tp4_diagnose_20261010/allocator-fix-20261010/test_cpu_allocator.py
```

Request measurements use `allocator_service_client.py SERVICE_RUN OUTPUT_DIR`.
Raw records preserve the warmup, ten measured requests and one profiled request
separately. No long-generation or multi-prompt accuracy evaluation was run.
