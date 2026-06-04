# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression tests for DSV4 attention backend kernel block sizes on CPU
(M3.2 part 10).

Background: GPU's ``DeepseekV4FlashMLASparseBackend`` and
``DeepseekV4IndexerBackend`` declare ``[256]`` because the FlashMLA sparse
fwd kernel and the indexer's compressor-state co-location both require
256-token pages. CPU goes through ``cpu_forward_decode`` /
``cpu_sparse_attn_prefill`` / ``cpu_sparse_attn_indexer_op`` (in
``vllm.models.deepseek_v4.cpu``) which index plain bf16 tensors by
``slot_mapping`` / ``block_table`` and have no special block-size
requirement. With the GPU constraint left in place,
``select_common_block_size(128, [V4_FLASHMLA_SPARSE, V4_INDEXER, ...])``
raises ``ValueError: No common block size for 128.`` because no divisor
of CPU's default ``block_size=128`` is divisible by 256.

Part 10 declares ``MultipleOf(16)`` on CPU for both backends. These tests
patch ``current_platform.is_cpu()`` to True (so the CPU branch is taken
even when the test runs on darwin/x86 dev machines) and assert:

1. The CPU branch returns ``[MultipleOf(16)]``.
2. The GPU branch still returns ``[256]``.
3. ``select_common_block_size(128, [V4_FLASHMLA_SPARSE, V4_INDEXER,
   DEEPSEEK_SPARSE_SWA, CPU_ATTN])`` returns 128 without raising.

These are static-import tests — they do not instantiate any backend, do
not allocate KV caches, and do not require torch.cuda. They run on
darwin in the dev loop.
"""

from __future__ import annotations

import pytest

from vllm.models.deepseek_v4.nvidia.flashmla import (
    DeepseekV4FlashMLASparseBackend,
)
from vllm.v1.attention.backend import MultipleOf
from vllm.v1.attention.backends.cpu_attn import CPUAttentionBackend
from vllm.v1.attention.backends.mla.indexer import DeepseekV4IndexerBackend
from vllm.v1.attention.backends.mla.sparse_swa import DeepseekSparseSWABackend
from vllm.v1.worker.utils import select_common_block_size

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]


def _force_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force ``current_platform.is_cpu()`` to True in both backend modules.

    The two backends do ``from vllm.platforms import current_platform`` at
    module import time, so the ``current_platform`` symbol is bound into
    each backend module's namespace. Patching the global registry would not
    reach those bound references; we patch the bound objects directly.
    """
    import vllm.models.deepseek_v4.nvidia.flashmla as fmla_sparse_mod
    import vllm.v1.attention.backends.mla.indexer as indexer_mod

    monkeypatch.setattr(
        fmla_sparse_mod.current_platform, "is_cpu", lambda: True
    )
    monkeypatch.setattr(indexer_mod.current_platform, "is_cpu", lambda: True)


def _force_non_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force ``current_platform.is_cpu()`` to False in both backend modules."""
    import vllm.models.deepseek_v4.nvidia.flashmla as fmla_sparse_mod
    import vllm.v1.attention.backends.mla.indexer as indexer_mod

    monkeypatch.setattr(
        fmla_sparse_mod.current_platform, "is_cpu", lambda: False
    )
    monkeypatch.setattr(indexer_mod.current_platform, "is_cpu", lambda: False)


def test_v4_flashmla_sparse_cpu_returns_multipleof16(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _force_cpu(monkeypatch)
    sizes = DeepseekV4FlashMLASparseBackend.get_supported_kernel_block_sizes()
    # MultipleOf has no __eq__, compare structurally.
    assert len(sizes) == 1
    assert isinstance(sizes[0], MultipleOf)
    assert sizes[0].base == 16


def test_v4_flashmla_sparse_gpu_returns_256(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _force_non_cpu(monkeypatch)
    sizes = DeepseekV4FlashMLASparseBackend.get_supported_kernel_block_sizes()
    assert sizes == [256], sizes


def test_v4_indexer_cpu_returns_multipleof16(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _force_cpu(monkeypatch)
    sizes = DeepseekV4IndexerBackend.get_supported_kernel_block_sizes()
    assert len(sizes) == 1
    assert isinstance(sizes[0], MultipleOf)
    assert sizes[0].base == 16


def test_v4_indexer_gpu_returns_256(monkeypatch: pytest.MonkeyPatch) -> None:
    _force_non_cpu(monkeypatch)
    sizes = DeepseekV4IndexerBackend.get_supported_kernel_block_sizes()
    assert sizes == [256], sizes


def test_select_common_block_size_128_with_v4_backends_on_cpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The end-to-end check that part 10 exists to fix.

    This is the call that originally raised
    ``ValueError: No common block size for 128.`` from
    ``vllm/v1/worker/utils.py:326`` during ``vllm serve --device cpu``
    on the ARM cluster.
    """
    _force_cpu(monkeypatch)
    backends = [
        DeepseekV4FlashMLASparseBackend,
        DeepseekV4IndexerBackend,
        DeepseekSparseSWABackend,
        CPUAttentionBackend,
    ]
    # CPU default block_size = 128 (platforms/cpu.py:122-123). Should be
    # supported by all four backends after part 10 (16 | 64 | 1 | 16 all
    # divide 128), so case 1 of select_common_block_size returns it
    # directly.
    assert select_common_block_size(128, backends) == 128


def test_select_common_block_size_128_without_part10_fix_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sanity check: if the CPU branch is *not* taken (i.e. the GPU
    [256] declarations remain active), the original error reproduces.

    This guards against future refactors that might silently drop the
    CPU branch and re-introduce the regression.
    """
    _force_non_cpu(monkeypatch)
    backends = [
        DeepseekV4FlashMLASparseBackend,
        DeepseekV4IndexerBackend,
        DeepseekSparseSWABackend,
        CPUAttentionBackend,
    ]
    with pytest.raises(ValueError, match="No common block size for 128"):
        select_common_block_size(128, backends)
