# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSpark registration guard for the CPU-isolated DeepSeek V4 package."""

import torch.nn as nn


class DSparkDeepseekV4ForCausalLM(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        raise NotImplementedError(
            "DeepSeek V4 DSpark drafting is not supported by the CPU backend; "
            "use the CPU MTP draft model or disable speculative decoding."
        )


__all__ = ["DSparkDeepseekV4ForCausalLM"]
