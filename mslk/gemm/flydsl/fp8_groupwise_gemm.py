# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

"""FP8 groupwise-scaled GEMM via FlyDSL.

Registers ``mslk::f8f8bf16_groupwise``, the plain (non-grouped) GEMM whose
weights are scaled per 128x128 block and whose activations are scaled per token
per 128 of K. CUDA implements it in CUTLASS; on ROCm it was a Triton kernel,
which this replaces.

It is served by the same kernel as the grouped ops, compiled for a single group
under the ``batched`` layout: that layout gives each group a fixed slab of rows,
so one group is one slab spanning all of M, which is a plain GEMM. The scale
layouts coincide too. Block scaling addresses scale_a as per-group blocks --
group g's block starts at ``m_start * scale_k`` and holds element
``(local_m, k_block)`` at ``local_m + k_block * M_g`` -- and at one group that is
``local_m + k_block * M``, which is exactly the ``[K//128, M]`` this op is
handed. Likewise ``[G, K//128, N//128]`` is ``[K//128, N//128]`` at G = 1.

Tensor contract:
  XQ      : [M, K]               FP8  -- activations
  WQ      : [N, K]               FP8  -- weights, already transposed
  x_scale : [K//128, M]          FP32 -- per (K-group-of-128, row)
  w_scale : [K//128, N//128]     FP32 -- per (K-group-of-128, N-group-of-128)
  Output  : [M, N]               BF16

  out[m, n] = sum_k XQ[m, k] * x_scale[k//128, m]
                  * WQ[n, k] * w_scale[k//128, n//128]
"""

import torch
from mslk.flydsl.common import is_flydsl_available
from mslk.gemm.flydsl import grouped_dispatch

_SCALE_BLOCK = grouped_dispatch.SCALE_BLOCK


def matmul_f8f8bf16_groupwise(
    XQ: torch.Tensor,
    WQ: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
) -> torch.Tensor:
    """FP8 groupwise-scaled GEMM -> BF16."""
    assert XQ.ndim == 2, f"XQ must be [M, K], got {XQ.shape}"
    assert WQ.ndim == 2, f"WQ must be [N, K], got {WQ.shape}"
    M, K = XQ.shape
    N, Kw = WQ.shape
    assert Kw == K, f"K mismatch: XQ K={K}, WQ K={Kw}"
    grouped_dispatch.assert_fp8_operands(XQ, WQ)
    # A scale covers a whole block, so N and K have to span whole ones: a
    # partial block at either end falls outside the count and misindexes the
    # scales from there on. The CUDA implementation fixes the same granularity.
    assert N % _SCALE_BLOCK == 0 and K % _SCALE_BLOCK == 0, (
        f"n ({N}) and k ({K}) must be multiples of the {_SCALE_BLOCK}-element "
        "scale block under block scaling"
    )
    scale_k, scale_n = K // _SCALE_BLOCK, N // _SCALE_BLOCK
    assert x_scale.numel() == scale_k * M, (
        f"x_scale must be [{scale_k}, {M}], got {tuple(x_scale.shape)}"
    )
    assert w_scale.numel() == scale_k * scale_n, (
        f"w_scale must be [{scale_k}, {scale_n}], got {tuple(w_scale.shape)}"
    )

    # One group owning every row, so the weights become the single-entry stack
    # the kernel indexes by group. The view is free on a contiguous tensor.
    return grouped_dispatch.dispatch(
        XQ,
        WQ.unsqueeze(0),
        x_scale,
        w_scale,
        grouped_dispatch.unused_group_meta(XQ.device),
        b_preshuffled=False,
        blockscale=True,
        layout="batched",
    )


if (
    is_flydsl_available()
    and torch.version.hip is not None
    and hasattr(torch.ops, "mslk")
):
    # FlyDSL supplies the ROCm implementation; the schema is declared in
    # csrc/gemm/gemm_ops.cpp and the fake impl in mslk/gemm/_meta.py. This
    # replaces the Triton registration, which mslk/gemm/__init__.py imports
    # first. Skip the op if its schema is missing, as in a python-only build,
    # and tolerate a repeat import rebinding it.
    if hasattr(torch.ops.mslk, "f8f8bf16_groupwise"):
        try:
            torch.library.impl("mslk::f8f8bf16_groupwise", "CUDA")(
                matmul_f8f8bf16_groupwise
            )
        except RuntimeError:
            pass
