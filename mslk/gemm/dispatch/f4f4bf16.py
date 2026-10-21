# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Which backend serves ``mslk::f4f4bf16`` (MXFP4 x MXFP4 -> BF16) on ROCm.

- ``mslk``: MSLK's Triton kernel (``mslk/gemm/triton/f4f4bf16.py``), the
  default and the fallback.
- ``aiter``: aiter's ``gemm_afp4wfp4`` (``mslk/gemm/aiter/f4f4bf16.py``), for
  the shapes ``tables/f4f4bf16.csv`` assigns to it. That table is written by
  ``bench/aiter/tune_f4f4bf16.py``.

aiter only computes standard MXFP4 (no global scale, 32-element scale blocks);
other calls go to the Triton path, which rejects them on ROCm.
"""

from typing import Optional

import torch
from mslk.aiter.routing import MSLK, OpRouting

OP: str = "f4f4bf16"
AITER: str = "aiter"
BACKENDS: tuple[str, ...] = (MSLK, AITER)

ROUTING: OpRouting = OpRouting(
    OP,
    "mslk.gemm.dispatch",
    "tables/f4f4bf16.csv",
    key_columns=("N", "K"),
    bucket_column="M_max",
)


def backend_for(
    XQ: torch.Tensor,
    WQ: torch.Tensor,
    global_scale: Optional[torch.Tensor] = None,
    mxfp4_block_size: int = 32,
) -> str:
    """The backend that serves ``f4f4bf16`` on these arguments."""
    if global_scale is not None or mxfp4_block_size != 32:
        return MSLK
    M, K = XQ.shape[0], XQ.shape[1] * 2
    N = WQ.shape[0]
    return ROUTING.backend(XQ.get_device(), (N, K), M)


def f4f4bf16(
    XQ: torch.Tensor,
    WQ: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    output: Optional[torch.Tensor] = None,
    global_scale: Optional[torch.Tensor] = None,
    mxfp4_block_size: int = 32,
) -> torch.Tensor:
    """``mslk::f4f4bf16`` on ROCm, served by the backend :func:`backend_for` picks."""
    # Backends are imported on first use: importing mslk.gemm must not pull in
    # Triton on CPU-only nodes, nor aiter anywhere.
    if backend_for(XQ, WQ, global_scale, mxfp4_block_size) == AITER:
        from mslk.gemm.aiter import f4f4bf16 as aiter_f4f4bf16

        return aiter_f4f4bf16.f4f4bf16(XQ, WQ, x_scale, w_scale, output)

    from mslk.gemm.triton import f4f4bf16 as triton_f4f4bf16

    return triton_f4f4bf16.f4f4bf16(
        XQ,
        WQ,
        x_scale,
        w_scale,
        output=output,
        global_scale=global_scale,
        mxfp4_block_size=mxfp4_block_size,
    )
