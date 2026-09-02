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

Two kernels serve it, chosen by architecture.

On gfx950 it is a kernel written for this op alone, built around instructions
that architecture introduced -- the wide ``f8f6f4`` MFMA and ``permlane32_swap``
-- and around the fact that a plain GEMM needs no group resolution at all. See
``mslk.flydsl.kernels.gemm.fp8_groupwise_wide_gemm``.

Everywhere else it is the same kernel as the grouped ops, compiled for a single
group under the ``batched`` layout: that layout gives each group a fixed slab of
rows, so one group is one slab spanning all of M, which is a plain GEMM. The
scale layouts coincide too. Block scaling addresses scale_a as per-group blocks
-- group g's block starts at ``m_start * scale_k`` and holds element
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

import functools
from collections.abc import Callable

import torch
from mslk.gemm.flydsl import grouped_dispatch
from mslk.utils.device import is_gfx942, is_gfx950

_SCALE_BLOCK = grouped_dispatch.SCALE_BLOCK


def _matmul_gfx942(XQ, WQ, x_scale, w_scale, M, N, K):
    """Serve the op from the grouped GEMM kernel, as one batched group."""
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


def _matmul_gfx950(XQ, WQ, x_scale, w_scale, M, N, K):
    """Serve the op from the kernel written for it.

    The tile and wave grid are left at the kernel's defaults until they are
    swept: which one wins is a per-shape question, and a hand-written ladder
    would only be a guess to unpick later.
    """
    from mslk.flydsl.jit import run_compiled
    from mslk.flydsl.kernels.gemm.fp8_groupwise_wide_gemm import (
        compile_groupwise_wide_gemm,
    )

    launcher = compile_groupwise_wide_gemm(n=N, k=K, tile_k=_SCALE_BLOCK)
    # The grid covers every row and column of the output, so nothing is left
    # unwritten and the buffer does not have to start zeroed.
    out = torch.empty((M, N), dtype=torch.bfloat16, device=XQ.device)
    # The kernel addresses the operands as flat byte buffers; FP8 is viewed as
    # int8 for the handoff, as the grouped path does.
    run_compiled(
        launcher,
        out,
        XQ.contiguous().view(torch.int8),
        WQ.contiguous().view(torch.int8),
        x_scale.contiguous(),
        w_scale.contiguous(),
        M,
        N,
        K,
        torch.cuda.current_stream(),
    )
    return out


@functools.lru_cache(maxsize=1)
def _kernel() -> Callable[..., torch.Tensor]:
    """Which kernel serves this op on the current GPU.

    Named architectures rather than "gfx950 or else": FlyDSL reports itself
    available on the RDNA parts too, which have no MFMA at all, and falling
    through to a kernel built on one would fail somewhere deep inside it.

    Resolved once rather than per call: the architecture cannot change within a
    process, and reading it costs a device-property lookup that dwarfs a tensor
    attribute access.
    """
    if is_gfx950():
        return _matmul_gfx950
    if is_gfx942():
        return _matmul_gfx942
    raise RuntimeError(
        "mslk::f8f8bf16_groupwise on ROCm is implemented for gfx942 and gfx950; "
        "this GPU is neither."
    )


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

    return _kernel()(XQ, WQ, x_scale, w_scale, M, N, K)


# This module deliberately does not register the op. FlyDSL owns it wherever it
# is opted into and Triton is the fallback, but only one CUDA implementation can
# win, so the choice is arbitrated in mslk/gemm/__init__.py on first call --
# which also keeps registration from importing FlyDSL.
