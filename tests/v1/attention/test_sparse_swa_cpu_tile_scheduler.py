# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression test for DSV4 SWA metadata builder ``build_tile_scheduler``
on CPU (M3.2 part 14).

Background
----------
After M3.2 parts 10-13 unblocked attention metadata + the first per-layer
``_fused_qnorm_rope_kv_insert``, the next chat-completions request on
ARM CPU TP=4 hit a new traceback in
``DeepseekSparseSWAMetadataBuilder.build()``:

    File "vllm/v1/attention/backends/mla/sparse_swa.py", line 432,
        in build_tile_scheduler
        out[layer_type] = get_mla_metadata()[0]
    File "vllm/v1/attention/ops/flashmla.py", line 83,
        in _raise_flashmla_unavailable
    RuntimeError: vllm._flashmla_C is not available, likely was not
        compiled due to insufficient nvcc version or a supported arch
        was not in the list of target arches to compile for.

``get_mla_metadata`` is a thin Python wrapper around
``torch.ops._C_flashmla.get_mla_metadata``; on CPU wheels the
``vllm._flashmla_C`` extension is never built, so the import-time stub
unconditionally raises.

The CPU forward-decode path (``DeepseekV4MLAAttention._forward_decode``
at ``deepseek_v4_attention.py:1077-1094``) calls ``cpu_forward_decode``,
which doesn't read ``tile_sched_swaonly`` / ``tile_sched_c4a`` /
``tile_sched_c128a`` at all — exactly mirroring ROCm's
``rocm_forward_decode_fallback`` (line 1095-1110). The ROCm branch
already had a ``current_platform.is_rocm()`` early-return at line 425;
this fix adds the matching ``is_cpu()`` clause.

We assert two things:

1. The CPU branch in ``build_tile_scheduler`` short-circuits to all-None
   without calling ``get_mla_metadata()``. Driven via a behavioural test
   that monkey-patches the module-bound ``current_platform`` and the
   ``get_mla_metadata`` import to trip-wires.
2. Source-level guard so the predicate stays correct even if the inner
   loop body changes shape later.

See also: ``project_dsv4_arm_cpu.md`` (M3.2-part14).
"""

from __future__ import annotations

import inspect
import re
import textwrap
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# Source-inspection guard
# ---------------------------------------------------------------------------


def _strip_comments(src: str) -> str:
    out_lines = []
    for line in src.splitlines():
        if line.lstrip().startswith("#"):
            continue
        idx = line.find("#")
        if idx >= 0:
            line = line[:idx]
        out_lines.append(line)
    return "\n".join(out_lines)


def _build_tile_scheduler_source() -> str:
    from vllm.v1.attention.backends.mla.sparse_swa import (
        DeepseekSparseSWAMetadataBuilder,
    )

    src = inspect.getsource(
        DeepseekSparseSWAMetadataBuilder.build_tile_scheduler
    )
    return _strip_comments(textwrap.dedent(src))


def test_build_tile_scheduler_short_circuits_on_cpu() -> None:
    """The CPU early-return predicate must be present in the function body."""
    src = _build_tile_scheduler_source()

    # Find the early-return ``if ...: return out``. Ripgrep-friendly: we
    # want the *first* such guard in the function (multi-line allowed).
    m = re.search(
        r"if\s+(.*?)\s*:\s*\n\s*return\s+out\b",
        src,
        re.DOTALL,
    )
    assert m is not None, (
        "build_tile_scheduler must contain an early-return guard "
        "before the get_mla_metadata loop."
    )
    predicate = m.group(1)

    # The guard MUST contain both is_rocm and is_cpu (alongside the
    # num_decode_tokens == 0 base case). We don't pin the exact spelling
    # so the guard can be reformatted/expanded.
    assert "current_platform.is_cpu()" in predicate, (
        "build_tile_scheduler must short-circuit on CPU. "
        "get_mla_metadata depends on vllm._flashmla_C which is not built "
        "on CPU wheels and raises RuntimeError. Predicate seen:\n"
        f"  {predicate!r}"
    )
    assert "current_platform.is_rocm()" in predicate, (
        "ROCm guard regressed — that path must keep its short-circuit "
        "(rocm_forward_decode_fallback also doesn't use tile_sched_*). "
        f"Predicate seen:\n  {predicate!r}"
    )
    assert "num_decode_tokens" in predicate, (
        "The 'no decode tokens this step' base case must remain — "
        "build_tile_scheduler can be called with num_decode_tokens=0 "
        f"on every platform. Predicate seen:\n  {predicate!r}"
    )


def test_build_tile_scheduler_does_not_import_get_mla_metadata_only_on_cpu(
) -> None:
    """``get_mla_metadata`` is still imported at module-top so the GPU path
    works; we only need to guarantee the call site isn't reached on CPU."""
    from vllm.v1.attention.backends.mla import sparse_swa

    # Module-top imports of get_mla_metadata are fine on CPU because
    # vllm/v1/attention/ops/flashmla.py replaces it with a stub that only
    # raises *when called* — confirm it's still importable on the test
    # platform (darwin), proving we don't need to lazy-import.
    assert hasattr(sparse_swa, "get_mla_metadata"), (
        "sparse_swa must keep its top-level get_mla_metadata import; the "
        "fix is gating the call site, not the import."
    )


# ---------------------------------------------------------------------------
# Behavioural guard: drive the function with a tripwire
# ---------------------------------------------------------------------------


def _make_minimal_builder():
    """Construct a ``DeepseekSparseSWAMetadataBuilder`` instance bypassing
    ``__init__`` so we can call ``build_tile_scheduler`` in isolation.
    Only the attributes that ``build_tile_scheduler`` reads matter:
    ``self._layer_types`` (a list[str]). Bind the unbound method onto a
    plain object to avoid touching the heavy dataclass init.
    """
    from vllm.v1.attention.backends.mla.sparse_swa import (
        _LAYER_TYPE_C4A,
        _LAYER_TYPE_C128A,
        _LAYER_TYPE_SWAONLY,
        DeepseekSparseSWAMetadataBuilder,
    )

    builder = DeepseekSparseSWAMetadataBuilder.__new__(
        DeepseekSparseSWAMetadataBuilder
    )
    builder._layer_types = [
        _LAYER_TYPE_SWAONLY,
        _LAYER_TYPE_C4A,
        _LAYER_TYPE_C128A,
    ]
    return builder


def test_build_tile_scheduler_returns_all_none_on_cpu_without_calling_flashmla(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end check that on CPU we don't hit ``get_mla_metadata``."""
    from vllm.v1.attention.backends.mla import sparse_swa
    from vllm.v1.attention.backends.mla.sparse_swa import (
        _LAYER_TYPE_C4A,
        _LAYER_TYPE_C128A,
        _LAYER_TYPE_SWAONLY,
    )

    # Module-bound ``current_platform`` (parts 2/3/10/11 followed the same
    # patching pattern; patching the global registry doesn't propagate
    # here because the backend imported it at module load).
    monkeypatch.setattr(sparse_swa.current_platform, "is_cpu", lambda: True)
    monkeypatch.setattr(sparse_swa.current_platform, "is_rocm", lambda: False)

    # Tripwire: if the function ever reaches ``get_mla_metadata`` on CPU,
    # surface a distinct exception type so we can attribute it cleanly.
    sentinel = MagicMock(
        side_effect=AssertionError(
            "get_mla_metadata must NOT be called when current_platform.is_cpu()"
        )
    )
    monkeypatch.setattr(sparse_swa, "get_mla_metadata", sentinel)

    builder = _make_minimal_builder()
    out = builder.build_tile_scheduler(num_decode_tokens=4)

    assert out == {
        _LAYER_TYPE_SWAONLY: None,
        _LAYER_TYPE_C4A: None,
        _LAYER_TYPE_C128A: None,
    }, (
        f"Expected all-None tile-scheduler dict on CPU, got {out!r}."
    )
    sentinel.assert_not_called()


def test_build_tile_scheduler_short_circuits_when_no_decode_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sanity: the num_decode_tokens==0 base case still short-circuits on
    every platform — covers the case where a step has prefill-only work."""
    from vllm.v1.attention.backends.mla import sparse_swa

    monkeypatch.setattr(sparse_swa.current_platform, "is_cpu", lambda: False)
    monkeypatch.setattr(sparse_swa.current_platform, "is_rocm", lambda: False)
    sentinel = MagicMock(
        side_effect=AssertionError(
            "get_mla_metadata must NOT be called when num_decode_tokens == 0"
        )
    )
    monkeypatch.setattr(sparse_swa, "get_mla_metadata", sentinel)

    builder = _make_minimal_builder()
    out = builder.build_tile_scheduler(num_decode_tokens=0)

    assert all(v is None for v in out.values())
    sentinel.assert_not_called()


def test_build_tile_scheduler_calls_flashmla_on_gpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Negative control: when neither short-circuit fires, the loop must
    actually call ``get_mla_metadata`` once per layer type. Guards
    against a future "always short-circuit" regression that would silently
    break the GPU path."""
    from vllm.v1.attention.backends.mla import sparse_swa

    monkeypatch.setattr(sparse_swa.current_platform, "is_cpu", lambda: False)
    monkeypatch.setattr(sparse_swa.current_platform, "is_rocm", lambda: False)

    fake_meta = ("FAKE_TILE_META", "FAKE_NUM_SPLITS")
    fake_get_mla_metadata = MagicMock(return_value=fake_meta)
    monkeypatch.setattr(sparse_swa, "get_mla_metadata", fake_get_mla_metadata)

    builder = _make_minimal_builder()
    out = builder.build_tile_scheduler(num_decode_tokens=4)

    # 3 layer types → 3 calls.
    assert fake_get_mla_metadata.call_count == 3, (
        f"Expected 3 get_mla_metadata calls (one per layer type), got "
        f"{fake_get_mla_metadata.call_count}"
    )
    # Each entry is the [0] element of the get_mla_metadata return tuple.
    assert all(v == "FAKE_TILE_META" for v in out.values())
