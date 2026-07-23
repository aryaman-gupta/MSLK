# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

"""FP8 groupwise-scaled grouped GEMM (preshuffled weights) via FlyDSL.

Registers ``mslk::f8f8bf16_groupwise_grouped_preshuffle``, the ROCm sibling of
``mslk::f8f8bf16_groupwise_grouped`` that consumes weights already in the MFMA
B-preshuffle layout (see ``mslk.quantize.shuffle.preshuffle_b_mfma``). Callers
shuffle weights once at load time; the op does no shuffling.

Tensor contract:
  XQ      : [TotalM, K]             FP8  -- all groups concatenated along M
  WQ      : [G, N, K]               FP8  -- per-group weights, MFMA-preshuffled
  x_scale : [K//128, TotalM]        FP32 -- per-token per-128K scales (transposed)
  w_scale : [G, K//128, N//128]     FP32 -- per-group per-block scales
  M_sizes : [G]                     int64 -- rows per group (sum to TotalM)
  Output  : [TotalM, N]             BF16
"""

import torch

from mslk.gemm.flydsl.grouped_gemm_tuning import get_tile_config
from mslk.utils.flydsl import is_flydsl_available, run_compiled

_OP_NAME = "mslk::f8f8bf16_groupwise_grouped_preshuffle"

# tile_m/tile_n/tile_k are now selected per-shape by get_tile_config; only the
# scale-block granularity is fixed.
_SCALE_BLOCK = 128

torch.library.define(
    _OP_NAME,
    "(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor M_sizes) -> Tensor",
)


@torch.library.impl(_OP_NAME, "Meta")
def _f8f8bf16_groupwise_grouped_preshuffle_meta(
    XQ: torch.Tensor,
    WQ: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    M_sizes: torch.Tensor,
) -> torch.Tensor:
    TotalM = XQ.shape[0]
    N = WQ.shape[1]
    return XQ.new_empty((TotalM, N), dtype=torch.bfloat16)


def _dispatch_grouped_gemm(
    XQ: torch.Tensor,
    WQ: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    M_sizes: torch.Tensor,
    *,
    b_preshuffled: bool,
) -> torch.Tensor:
    """Shared dispatch for both grouped ops. WQ is already in the layout the
    variant expects (MFMA-preshuffled if b_preshuffled else plain [G,N,K]).

    Selects (tile_m, tile_n, tile_k) from the static tuning table
    (grouped_gemm_tuning.get_tile_config), then builds the per-tile dispatch map,
    compiles, and launches. The tuned tile_m is used CONSISTENTLY for the
    tile-map / grid extent AND the kernel compile (they must agree).
    """
    from mslk.flydsl.kernels.gemm.grouped_gemm_blockscale_contiguous import (
        compile_grouped_gemm_blockscale_contiguous,
    )

    assert XQ.ndim == 2, f"XQ must be [TotalM, K], got {XQ.shape}"
    assert WQ.ndim == 3, f"WQ must be [G, N, K], got {WQ.shape}"
    TotalM, K = XQ.shape
    G, N, Kw = WQ.shape
    assert Kw == K, f"K mismatch: XQ K={K}, WQ K={Kw}"

    tile_m, tile_n, tile_k = get_tile_config(
        b_preshuffled=b_preshuffled, total_m=TotalM, n=N, k=K, g=G
    )
    assert N % tile_n == 0, f"N={N} must be a multiple of tile_n={tile_n}"
    assert K % tile_k == 0, f"K={K} must be a multiple of tile_k={tile_k}"

    output = torch.empty((TotalM, N), dtype=torch.bfloat16, device=XQ.device)
    if TotalM == 0 or N == 0 or K == 0 or G == 0:
        return output

    # Grid M-extent: a host-known upper bound (each group wastes at most one
    # partial tile). Computed from static shapes, never from M_sizes contents, so
    # dispatch stays sync-free (no .item() device->host stall) in eager mode and
    # remains valid under CUDA-graph capture. The kernel resolves group ownership
    # per M-tile directly from M_sizes; surplus tiles (bx >= actual tile count)
    # match no group and self-skip, so overcounting only costs a few no-op blocks.
    num_m_tiles = TotalM // tile_m + G

    # The kernel reads M_sizes as int32 (buffer_load i32 idiom). Cast once on the
    # host -- a single tiny kernel, vs the ~8-kernel host tile-map this replaced.
    m_sizes_i32 = M_sizes.to(torch.int32)

    launcher = compile_grouped_gemm_blockscale_contiguous(
        n=N,
        k=K,
        num_groups=G,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        scale_block_k=_SCALE_BLOCK,
        scale_block_n=_SCALE_BLOCK,
        out_dtype="bf16",
        b_preshuffled=b_preshuffled,
    )

    # Kernel buffer resources use raw byte offsets, so pass flat tensors; FP8 is
    # viewed as int8 for the DLPack handoff.
    run_compiled(
        launcher,
        output.view(-1),
        XQ.contiguous().view(-1).view(torch.int8),
        WQ.contiguous().view(-1).view(torch.int8),
        x_scale.contiguous().view(-1),
        w_scale.contiguous().view(-1),
        m_sizes_i32,
        TotalM,
        N,
        K,
        G,
        num_m_tiles,
        torch.cuda.current_stream(),
    )
    return output


def matmul_f8f8bf16_groupwise_grouped_preshuffle(
    XQ: torch.Tensor,
    WQ: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    M_sizes: torch.Tensor,
) -> torch.Tensor:
    """Preshuffled-B grouped groupwise FP8 GEMM (WQ already MFMA-preshuffled)."""
    return _dispatch_grouped_gemm(
        XQ, WQ, x_scale, w_scale, M_sizes, b_preshuffled=True
    )


def matmul_f8f8bf16_groupwise_grouped(
    XQ: torch.Tensor,
    WQ: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    M_sizes: torch.Tensor,
) -> torch.Tensor:
    """Plain (non-preshuffled) grouped groupwise FP8 GEMM via FlyDSL.

    Same contract as the preshuffle sibling but WQ is plain row-major [G, N, K]
    (not MFMA-preshuffled) — the drop-in FlyDSL replacement for the Triton impl
    of ``mslk::f8f8bf16_groupwise_grouped``. Uses the unified kernel with
    ``b_preshuffled=False`` (B staged HBM->LDS->registers).
    """
    return _dispatch_grouped_gemm(
        XQ, WQ, x_scale, w_scale, M_sizes, b_preshuffled=False
    )


if is_flydsl_available():

    @torch.library.impl(_OP_NAME, "CUDA")
    def _f8f8bf16_groupwise_grouped_preshuffle_cuda(
        XQ: torch.Tensor,
        WQ: torch.Tensor,
        x_scale: torch.Tensor,
        w_scale: torch.Tensor,
        M_sizes: torch.Tensor,
    ) -> torch.Tensor:
        return matmul_f8f8bf16_groupwise_grouped_preshuffle(
            XQ, WQ, x_scale, w_scale, M_sizes
        )

    # Plain op: FlyDSL is the ROCm impl of mslk::f8f8bf16_groupwise_grouped
    # (C++ schema in gemm_ops.cpp; the Triton impl is being retired). Guard the
    # registration so it no-ops if the schema/op isn't present or already bound.
    if torch.version.hip is not None and hasattr(torch.ops, "mslk"):
        if hasattr(torch.ops.mslk, "f8f8bf16_groupwise_grouped"):
            try:

                @torch.library.impl("mslk::f8f8bf16_groupwise_grouped", "CUDA")
                def _f8f8bf16_groupwise_grouped_cuda(
                    XQ: torch.Tensor,
                    WQ: torch.Tensor,
                    x_scale: torch.Tensor,
                    w_scale: torch.Tensor,
                    M_sizes: torch.Tensor,
                ) -> torch.Tensor:
                    return matmul_f8f8bf16_groupwise_grouped(
                        XQ, WQ, x_scale, w_scale, M_sizes
                    )

            except RuntimeError:
                pass  # already registered
