# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for CPU unquantized GEMM dispatch behavior."""

import sys
import types

import pytest
import torch

from vllm.model_executor.kernels.linear.scaled_mm.cpu import (
    CPUInt8ScaledMMLinearKernel,
)
from vllm.model_executor.layers import utils
from vllm.platforms import current_platform
from vllm.platforms.interface import CpuArchEnum


@pytest.fixture(scope="module")
def _mock_zentorch_linear_unary():
    """Register a mock zentorch_linear_unary op when zentorch is not installed.

    Allows the dispatch tests to run in CI without a real zentorch build.
    Skips registration when zentorch is already available.
    """
    if hasattr(torch.ops.zentorch, "zentorch_linear_unary"):
        yield
        return

    lib_def = torch.library.Library("zentorch", "DEF")
    lib_def.define(
        "zentorch_linear_unary("
        "Tensor input, "
        "Tensor weight, "
        "Tensor? bias, "
        "bool is_weight_prepacked=False"
        ") -> Tensor"
    )

    lib_impl = torch.library.Library("zentorch", "IMPL", "CPU")
    lib_impl.impl(
        "zentorch_linear_unary",
        lambda input, weight, bias, is_weight_prepacked=False: (
            torch.nn.functional.linear(input, weight, bias)
        ),
    )

    yield

    lib_impl._destroy()
    lib_def._destroy()


@pytest.mark.usefixtures("_mock_zentorch_linear_unary")
def test_dispatch_cpu_unquantized_gemm_uses_zentorch_on_zen(monkeypatch):
    monkeypatch.setattr(current_platform, "is_zen_cpu", lambda: True)

    layer = torch.nn.Linear(16, 8, bias=True)
    x = torch.randn(4, 16)
    expected = torch.nn.functional.linear(x, layer.weight, layer.bias)

    utils.dispatch_cpu_unquantized_gemm(layer, remove_weight=False)
    output = layer.cpu_linear(x, layer.weight, layer.bias)

    torch.testing.assert_close(output, expected)


@pytest.mark.usefixtures("_mock_zentorch_linear_unary")
def test_dispatch_cpu_unquantized_gemm_zen_remove_weight(monkeypatch):
    monkeypatch.setattr(current_platform, "is_zen_cpu", lambda: True)

    layer = torch.nn.Linear(16, 8, bias=True)
    utils.dispatch_cpu_unquantized_gemm(layer, remove_weight=True)

    assert layer.weight.numel() == 0


@pytest.mark.usefixtures("_mock_zentorch_linear_unary")
def test_dispatch_cpu_unquantized_gemm_logs_zentorch_dispatch(monkeypatch):
    monkeypatch.setattr(current_platform, "is_zen_cpu", lambda: True)
    expected_prepacked = bool(utils.envs.VLLM_ZENTORCH_WEIGHT_PREPACK) and hasattr(
        torch.ops.zentorch, "zentorch_weight_prepack_for_linear"
    )

    log_calls = []
    monkeypatch.setattr(
        utils.logger, "debug_once", lambda *args: log_calls.append(args)
    )

    layer = torch.nn.Linear(16, 8, bias=True)
    utils.dispatch_cpu_unquantized_gemm(layer, remove_weight=False)

    assert log_calls == [
        (
            "CPU unquantized GEMM dispatch: using zentorch_linear_unary (prepacked=%s)",
            expected_prepacked,
        )
    ]


def test_required_fused_cpp_bf16_linear_prepares_once(monkeypatch):
    calls = {"prepare": 0, "linear": 0}

    class FakeBF16Linear:
        _supports_bf16_linear = True

        @staticmethod
        def prepare(weight):
            calls["prepare"] += 1
            return weight.detach().clone()

        @staticmethod
        def linear(x, prepared, *, out_dtype, nthreads):
            calls["linear"] += 1
            assert out_dtype == torch.bfloat16
            assert nthreads == torch.get_num_threads()
            return torch.nn.functional.linear(x, prepared).to(out_dtype)

    fake_fused_cpp = types.ModuleType("fused_cpp")
    fake_fused_cpp.bf16_linear = FakeBF16Linear
    monkeypatch.setitem(sys.modules, "fused_cpp", fake_fused_cpp)
    monkeypatch.setattr(
        current_platform, "get_cpu_architecture", lambda: CpuArchEnum.ARM
    )

    layer = torch.nn.Linear(16, 8, bias=True, dtype=torch.bfloat16)
    layer._cpu_fused_cpp_linear_required = True
    x = torch.randn(4, 16, dtype=torch.bfloat16)
    # fused_cpp returns BF16 before the optional bias add, so compare with the
    # same two-stage rounding rather than Torch's fused linear+bias epilogue.
    expected = torch.nn.functional.linear(x, layer.weight) + layer.bias

    utils.dispatch_cpu_unquantized_gemm(layer, remove_weight=True)
    assert layer.weight.numel() == 0
    assert calls["prepare"] == 1

    output0 = layer.cpu_linear(x, layer.weight, layer.bias)
    output1 = layer.cpu_linear(x, layer.weight, layer.bias)
    torch.testing.assert_close(output0, expected)
    torch.testing.assert_close(output1, expected)
    assert calls == {"prepare": 1, "linear": 2}


def test_required_fused_cpp_bf16_linear_fails_closed(monkeypatch):
    fake_fused_cpp = types.ModuleType("fused_cpp")
    fake_fused_cpp.bf16_linear = types.SimpleNamespace(
        _supports_bf16_linear=False
    )
    monkeypatch.setitem(sys.modules, "fused_cpp", fake_fused_cpp)
    monkeypatch.setattr(
        current_platform, "get_cpu_architecture", lambda: CpuArchEnum.ARM
    )

    layer = torch.nn.Linear(16, 8, bias=False, dtype=torch.bfloat16)
    layer._cpu_fused_cpp_linear_required = True

    with pytest.raises(RuntimeError, match="C\\+\\+ backend is unavailable"):
        utils.dispatch_cpu_unquantized_gemm(layer, remove_weight=True)


def test_unmarked_linear_keeps_upstream_dispatch(monkeypatch):
    monkeypatch.setattr(current_platform, "is_zen_cpu", lambda: False)
    monkeypatch.setattr(utils, "check_cpu_sgl_kernel", lambda *args: False)
    monkeypatch.setattr(utils.ops, "_supports_onednn", False)

    layer = torch.nn.Linear(16, 8, bias=False, dtype=torch.bfloat16)
    utils.dispatch_cpu_unquantized_gemm(layer, remove_weight=False)

    assert layer.cpu_linear is not None
    assert not hasattr(layer, "_cpu_fused_cpp_prepared_weight")


def test_mtp_int8_linear_dequantizes_once_to_bf16():
    layer = torch.nn.Module()
    int8_weight = torch.tensor([[2, -4], [3, 1]], dtype=torch.int8)
    scale = torch.tensor([0.5, 0.25], dtype=torch.float32)
    layer.register_parameter(
        "weight", torch.nn.Parameter(int8_weight, requires_grad=False)
    )
    layer.register_parameter(
        "weight_scale", torch.nn.Parameter(scale, requires_grad=False)
    )
    layer._cpu_int8_dequantize_to_bf16 = True

    kernel = object.__new__(CPUInt8ScaledMMLinearKernel)
    kernel.layer_param_names = (
        "weight",
        "weight_scale",
        "input_scale",
        "input_zero_point",
        "azp_adj",
    )
    kernel.process_weights_after_loading(layer)

    expected_weight = (int8_weight.float() * scale[:, None]).to(torch.bfloat16)
    torch.testing.assert_close(layer.weight, expected_weight)
    x = torch.tensor([[1.0, 2.0]], dtype=torch.bfloat16)
    output = kernel.apply_weights(layer, x)
    torch.testing.assert_close(output, torch.nn.functional.linear(x, expected_weight))
