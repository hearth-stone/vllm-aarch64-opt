# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression test for DSV4 SWA metadata builder ``pin_memory()`` on CPU
(M3.2 part 11).

Background: ``DeepseekSparseSWAMetadataBuilder.build()`` previously did

    x = torch.repeat_interleave(torch.arange(num_reqs), query_lens).pin_memory()

unconditionally. ``pin_memory()`` requires a CUDA host allocator; on
CPU-only ARM builds (the deployment target) PyTorch raises::

    RuntimeError: Cannot access accelerator device when none is available.

This is the first error the engine hits on the ARM cluster after parts
1-10 unblock model load + KV cache init: the very first
``execute_model -> _build_attention_metadata -> SWA builder.build()``
call dies before any token is produced. The fix mirrors
``deepseek_compressor.py:115-119`` (the M2a CPU fallback for the same
pattern): gate ``pin_memory()`` on ``not current_platform.is_cpu()``.
The follow-up ``copy_(x, non_blocking=True)`` degenerates to a
synchronous CPU memcpy — host->device-async semantics are a no-op on a
single-device CPU runtime.

We use a source-inspection test rather than driving ``build()``
end-to-end because the builder requires a fully wired ``vllm_config``
(scheduler/speculative/hf configs), a SlidingWindowMLASpec instance, a
real device, and four pre-allocated workspace buffers — instantiating
the surrounding scaffold for one line of behavior is disproportionate.
The check follows the same style as parts 2/3/8.
"""

from __future__ import annotations

import inspect
import re

import pytest

from vllm.v1.attention.backends.mla import sparse_swa


pytestmark = pytest.mark.cpu_test


def _build_method_source() -> str:
    """Source of ``DeepseekSparseSWAMetadataBuilder.build`` with comment-
    only lines stripped, so regex matches can't be fooled by an
    explanatory ``# comment`` quoting the unsafe pattern.
    """
    src = inspect.getsource(sparse_swa.DeepseekSparseSWAMetadataBuilder.build)
    return "\n".join(
        line for line in src.splitlines() if not line.lstrip().startswith("#")
    )


def test_sparse_swa_build_does_not_call_pin_memory_unconditionally() -> None:
    """The bare ``.pin_memory()`` chain on the ``torch.repeat_interleave(
    torch.arange(num_reqs), query_lens)`` expression must be gone — that
    is the exact pattern that crashes on CPU-only PyTorch builds.

    We allow ``pin_memory()`` to remain in the file (e.g. inside an
    ``if not current_platform.is_cpu():`` branch); we only forbid the
    fused chain ``...query_lens).pin_memory()`` at statement level
    because that's the unguarded form.
    """
    code = _build_method_source()
    # The exact pre-fix expression contained a nested call:
    #   torch.repeat_interleave(torch.arange(num_reqs), query_lens).pin_memory()
    # so a naive ``[^)]*`` would stop at the inner ``)``. Match more
    # permissively: any ``...query_lens).pin_memory()`` chain at the end
    # of the original buggy line is the regression we want to catch.
    assert not re.search(r"query_lens\)\.pin_memory\(\)", code), (
        "DeepseekSparseSWAMetadataBuilder.build() still chains "
        "``.pin_memory()`` directly onto ``torch.repeat_interleave(...)``; "
        "this raises ``RuntimeError: Cannot access accelerator device when "
        "none is available.`` on CPU-only PyTorch (the ARM deployment "
        "target). Gate the call on ``not current_platform.is_cpu()`` like "
        "deepseek_compressor.py:115-119 does."
    )


def test_sparse_swa_build_pin_memory_is_cpu_guarded() -> None:
    """The CPU guard must exist around any remaining ``pin_memory()``
    call inside ``build()``. We require both:

    * a ``current_platform.is_cpu()`` reference (the gate predicate)
    * a ``pin_memory()`` call somewhere in the method body (otherwise
      the GPU fast path is silently dropped, hurting GPU performance)
    """
    code = _build_method_source()

    assert "current_platform.is_cpu()" in code, (
        "DeepseekSparseSWAMetadataBuilder.build() should branch on "
        "``current_platform.is_cpu()`` to skip pin_memory() on CPU."
    )
    assert "pin_memory()" in code, (
        "DeepseekSparseSWAMetadataBuilder.build() lost its GPU "
        "``pin_memory()`` fast path entirely; CPU should be the only "
        "branch that skips it."
    )


def test_sparse_swa_module_imports_current_platform() -> None:
    """The CPU guard depends on ``current_platform`` being import-bound
    at module top level (not lazy-imported inside the method).

    This is also the symbol our regression tests in
    ``test_dsv4_cpu_block_sizes.py`` patch via
    ``monkeypatch.setattr(sparse_swa.current_platform, "is_cpu", ...)``;
    losing the top-level import would break that mock pattern.
    """
    assert hasattr(sparse_swa, "current_platform"), (
        "vllm.v1.attention.backends.mla.sparse_swa should ``from "
        "vllm.platforms import current_platform`` at module top level."
    )
