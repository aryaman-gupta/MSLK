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

from mslk.utils.flydsl import is_flydsl_available, run_compiled

_OP_NAME = "mslk::f8f8bf16_groupwise_grouped_preshuffle"

_TILE_M = 128
_TILE_N = 128
_TILE_K = 128
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


def _build_tile_map(
    M_sizes: torch.Tensor, tile_m: int, total_m: int, num_m_tiles_bound: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build the per-tile dispatch arrays consumed by the kernel.

    Returns ``(tile_group, tile_row_start, tile_row_limit)``, each int32 of
    length ``num_m_tiles_bound``. Entry ``t`` maps output M-tile ``t`` to its
    group id, its first global row, and its group's exclusive row end. Tiles
    past the actual tile count are marked ``tile_group = -1`` so the kernel
    skips them (the grid is launched to a host-known upper bound).

    Built with fixed-shape tensor ops (no ``.item()``) so it is safe to record
    into a CUDA graph and regenerate on replay from updated ``M_sizes``.
    """
    device = M_sizes.device
    G = M_sizes.shape[0]
    tiles_per_group = (M_sizes + (tile_m - 1)) // tile_m  # [G] int64
    m_starts = M_sizes.cumsum(0) - M_sizes  # [G] exclusive group row start
    tile_starts = tiles_per_group.cumsum(0) - tiles_per_group  # [G] first tile per group
    num_m_tiles = tiles_per_group.sum()  # scalar tensor (device)

    t = torch.arange(num_m_tiles_bound, device=device)  # [bound]
    # group id per tile = number of group-starts at or before t, minus 1.
    # searchsorted on the inclusive tile-end prefix gives the owning group.
    tile_ends = tile_starts + tiles_per_group  # [G] exclusive tile end per group
    tile_group = torch.searchsorted(tile_ends, t, right=True).to(torch.int32)  # [bound]

    # Clamp group index for gathering (surplus tiles read group 0, then masked).
    g_idx = tile_group.to(torch.long).clamp(max=G - 1)
    local_tile = t - tile_starts[g_idx]
    row_start = m_starts[g_idx] + local_tile * tile_m
    row_limit = m_starts[g_idx] + M_sizes[g_idx]

    # Mark surplus tiles (t >= num_m_tiles) as no-op.
    valid = t < num_m_tiles
    tile_group = torch.where(valid, tile_group, torch.full_like(tile_group, -1))
    tile_row_start = torch.where(
        valid, row_start, torch.zeros_like(row_start)
    ).to(torch.int32)
    tile_row_limit = torch.where(
        valid, row_limit, torch.zeros_like(row_limit)
    ).to(torch.int32)
    return tile_group, tile_row_start, tile_row_limit


def matmul_f8f8bf16_groupwise_grouped_preshuffle(
    XQ: torch.Tensor,
    WQ: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    M_sizes: torch.Tensor,
) -> torch.Tensor:
    from mslk.flydsl.kernels.gemm.grouped_gemm_blockscale_contiguous import (
        compile_grouped_gemm_blockscale_contiguous,
    )

    assert XQ.ndim == 2, f"XQ must be [TotalM, K], got {XQ.shape}"
    assert WQ.ndim == 3, f"WQ must be [G, N, K], got {WQ.shape}"
    TotalM, K = XQ.shape
    G, N, Kw = WQ.shape
    assert Kw == K, f"K mismatch: XQ K={K}, WQ K={Kw}"
    assert N % _TILE_N == 0, f"N={N} must be a multiple of tile_n={_TILE_N}"
    assert K % _TILE_K == 0, f"K={K} must be a multiple of tile_k={_TILE_K}"

    output = torch.empty((TotalM, N), dtype=torch.bfloat16, device=XQ.device)
    if TotalM == 0 or N == 0 or K == 0 or G == 0:
        return output

    # Grid M-extent: exact tile count when eager, host-known upper bound under
    # CUDA-graph capture (each group wastes at most one partial tile).
    upper_bound = TotalM // _TILE_M + G
    if torch.cuda.is_current_stream_capturing():
        num_m_tiles = upper_bound
    else:
        tiles_per_group = (M_sizes + (_TILE_M - 1)) // _TILE_M
        num_m_tiles = int(tiles_per_group.sum().item())

    tile_group, tile_row_start, tile_row_limit = _build_tile_map(
        M_sizes, _TILE_M, TotalM, num_m_tiles
    )

    launcher = compile_grouped_gemm_blockscale_contiguous(
        n=N,
        k=K,
        num_groups=G,
        tile_m=_TILE_M,
        tile_n=_TILE_N,
        tile_k=_TILE_K,
        scale_block_k=_SCALE_BLOCK,
        scale_block_n=_SCALE_BLOCK,
        out_dtype="bf16",
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
        tile_group,
        tile_row_start,
        tile_row_limit,
        TotalM,
        N,
        K,
        G,
        num_m_tiles,
        torch.cuda.current_stream(),
    )
    return output


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
    from mslk.flydsl.kernels.gemm.grouped_gemm_blockscale_contiguous import (
        compile_grouped_gemm_blockscale_contiguous,
    )

    assert XQ.ndim == 2, f"XQ must be [TotalM, K], got {XQ.shape}"
    assert WQ.ndim == 3, f"WQ must be [G, N, K], got {WQ.shape}"
    TotalM, K = XQ.shape
    G, N, Kw = WQ.shape
    assert Kw == K, f"K mismatch: XQ K={K}, WQ K={Kw}"
    assert N % _TILE_N == 0, f"N={N} must be a multiple of tile_n={_TILE_N}"
    assert K % _TILE_K == 0, f"K={K} must be a multiple of tile_k={_TILE_K}"

    output = torch.empty((TotalM, N), dtype=torch.bfloat16, device=XQ.device)
    if TotalM == 0 or N == 0 or K == 0 or G == 0:
        return output

    upper_bound = TotalM // _TILE_M + G
    if torch.cuda.is_current_stream_capturing():
        num_m_tiles = upper_bound
    else:
        tiles_per_group = (M_sizes + (_TILE_M - 1)) // _TILE_M
        num_m_tiles = int(tiles_per_group.sum().item())

    tile_group, tile_row_start, tile_row_limit = _build_tile_map(
        M_sizes, _TILE_M, TotalM, num_m_tiles
    )

    launcher = compile_grouped_gemm_blockscale_contiguous(
        n=N,
        k=K,
        num_groups=G,
        tile_m=_TILE_M,
        tile_n=_TILE_N,
        tile_k=_TILE_K,
        scale_block_k=_SCALE_BLOCK,
        scale_block_n=_SCALE_BLOCK,
        out_dtype="bf16",
        b_preshuffled=False,
    )

    # WQ is plain [G, N, K] — passed as-is (no preshuffle). FP8 viewed as int8.
    run_compiled(
        launcher,
        output.view(-1),
        XQ.contiguous().view(-1).view(torch.int8),
        WQ.contiguous().view(-1).view(torch.int8),
        x_scale.contiguous().view(-1),
        w_scale.contiguous().view(-1),
        tile_group,
        tile_row_start,
        tile_row_limit,
        TotalM,
        N,
        K,
        G,
        num_m_tiles,
        torch.cuda.current_stream(),
    )
    return output


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
