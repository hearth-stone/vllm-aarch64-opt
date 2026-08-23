# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU implementation of the DeepSeek V4 model family.

This package is deliberately isolated from the CUDA, ROCm, XPU, Triton, and
TileLang implementations.  Importing DeepSeek V4 on a CPU host must therefore
remain possible in a minimal Torch + fused_cpp installation.
"""

from .dspark import DSparkDeepseekV4ForCausalLM
from .model import DeepseekV4ForCausalLM
from .mtp import DeepSeekV4MTP

__all__ = [
    "DSparkDeepseekV4ForCausalLM",
    "DeepSeekV4MTP",
    "DeepseekV4ForCausalLM",
]
